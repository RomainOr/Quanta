from quanta_layer import QuantaLayer

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

def createModel(pInput, outputSize, layerToTranfer, sourceModel=None, augmentData=False):

    outputOfSourceLayers = []
    if sourceModel is not None:
        outputOfSourceLayers = [sourceModel.layers[i].output \
                for i in range(len(sourceModel.layers))   \
                if type(sourceModel.layers[i]).__name__ in ['Conv2D', 'Dense']]
        
    x = tf.keras.layers.Rescaling(1.0 / 255)(pInput)

    if sourceModel is not None and augmentData:
        x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")(x)
        x = tf.keras.layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1)(x)
        x = tf.keras.layers.RandomRotation(
                factor=0.1)(x)
        x = tf.keras.layers.RandomZoom(
                height_factor=0.1,
                width_factor=None)(x)

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

def compileModels(input_shape, targetTask, layerToTranfer, optimizer, loss, metrics, augmentData):

    print("Building source and target models : start")

    inputSource = tf.keras.Input(input_shape, name='inputSource')
    inputTarget = tf.keras.Input(input_shape, name='inputTarget')
    
    sourceModel = tf.keras.Model(
        inputs=inputSource,
        outputs=createModel(
            pInput=inputSource,
            outputSize=10,
            layerToTranfer=layerToTranfer)
        )
    sourceModel._name = "source"

    sourceModelCopy = tf.keras.Model(
        inputs=inputTarget,
        outputs=createModel(
            pInput=inputTarget,
            outputSize=10,
            layerToTranfer=layerToTranfer)
        )
    sourceModelCopy._name = "source_copy"
    for l in sourceModelCopy.layers: l.trainable=False

    if targetTask == 'cifar100':
        outputSize = 100
    else :
        outputSize = 10

    targetModel  = tf.keras.Model(
        inputs=inputTarget,
        outputs=createModel(
            pInput=inputTarget,
            outputSize=outputSize,
            layerToTranfer=layerToTranfer,
            sourceModel=sourceModelCopy,
            augmentData=augmentData)
        )
    targetModel._name = "target"

    sourceModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    targetModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print("Building source and target models : done\n")
    
    return sourceModel, sourceModelCopy, targetModel