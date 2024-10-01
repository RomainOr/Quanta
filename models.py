import sys
from QuantaLayer import QuantaLayer

import tensorflow as tf

#################################################
############# Convblock definition ##############
#################################################

def convBlock(pInput, size, layerToTranfer=None, outputOfSourceLayers=[]):
    x = pInput
    
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (layerToTranfer is not None and layerToTranfer%2==0 and outputOfSourceLayers != []):
        x = QuantaLayer('elu')([outputOfSourceLayers[0], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (layerToTranfer is not None and layerToTranfer%2==1 and outputOfSourceLayers != []):
        x = QuantaLayer('elu')([outputOfSourceLayers[1], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return x


#################################################
####### Create full model using convblocks ######
#################################################

def createModel(pInput, outputSize, layerToTranfer, sourceModel=None):
    outputOfSourceLayers = []
    if (sourceModel != None):
        outputOfSourceLayers = [sourceModel.layers[i].output \
                for i in range(len(sourceModel.layers))   \
                if type(sourceModel.layers[i]).__name__ in ['Conv2D', 'Dense']]
    x = pInput

    if (outputOfSourceLayers != [] and (layerToTranfer == 0 or layerToTranfer == 1)):
        x = convBlock(x, 64, layerToTranfer, [outputOfSourceLayers[0], outputOfSourceLayers[1]])
    else:
        x = convBlock(x, 64)

    if (outputOfSourceLayers != [] and (layerToTranfer == 2 or layerToTranfer == 3)):
        x = convBlock(x, 128, layerToTranfer, [outputOfSourceLayers[2], outputOfSourceLayers[3]])
    else:
        x = convBlock(x, 128)

    if (outputOfSourceLayers != [] and (layerToTranfer == 4 or layerToTranfer == 5)):
        x = convBlock(x, 512, layerToTranfer, [outputOfSourceLayers[4], outputOfSourceLayers[5]])
    x = convBlock(x, 512)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (outputOfSourceLayers != [] and layerToTranfer == 6):
        x = QuantaLayer('elu')([outputOfSourceLayers[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)
    
    return x


#################################################
####### Configure the models for training #######
#################################################

def compileModels(targetTask, layerToTranfer, optimizer, loss, metrics):

    print("Building source and target models : start")

    inputSource = tf.keras.Input([32, 32, 3], name='inputSource')
    inputTarget = tf.keras.Input([32, 32, 3], name='inputTarget')
    
    NNSource     = tf.keras.Model(
        inputs=inputSource,
        outputs=createModel(
            pInput=inputSource,
            outputSize=10,
            layerToTranfer=layerToTranfer)
        )

    NNSourceCopy = tf.keras.Model(
        inputs=inputTarget,
        outputs=createModel(
            pInput=inputTarget,
            outputSize=10,
            layerToTranfer=layerToTranfer)
        )
    for l in NNSourceCopy.layers: l.trainable=False

    if targetTask == 'cifar100':
        outputSize = 100
    else :
        outputSize = 10

    NNTarget  = tf.keras.Model(
        inputs=inputTarget,
        outputs=createModel(
            pInput=inputTarget,
            outputSize=outputSize,
            layerToTranfer=layerToTranfer,
            sourceModel=NNSourceCopy)
        )

    NNSource.compile(optimizer, loss, metrics)
    NNTarget.compile(optimizer, loss, metrics)

    NotLoadSource  = True
    NotLoadTarget  = True
    
    print("Building source and target models : done\n")
    
    return ((NNSource, NNSourceCopy, NNTarget),\
              (NotLoadSource, NotLoadTarget))