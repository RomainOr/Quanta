from QuantaLayer import QuantaLayer

import tensorflow as tf

####################################################################
## Convblock definition
####################################################################
def convBlock(pInput, size, transferedLayer, transferredLayers=[]):
    x = pInput
    
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (transferedLayer%2==0 and transferredLayers != []):
        x = QuantaLayer('elu')([transferredLayers[0], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (transferedLayer%2==1 and transferredLayers != []):
        x = QuantaLayer('elu')([transferredLayers[1], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return x

####################################################################
## Create full model using convblocks
####################################################################
def createModel(placeholder, outputSize, transferedLayer, sourceModel=None):
    transferredLayers = []
    if (sourceModel != None):
        transferredLayers = [sourceModel.layers[i].output \
                for i in range(len(sourceModel.layers))   \
                if type(sourceModel.layers[i]).__name__ in ['Conv2D', 'Dense']]
    x = placeholder

    if (transferredLayers != [] and (transferedLayer == 0 or transferedLayer == 1)):
        x = convBlock(x, 64, transferedLayer, [transferredLayers[0], transferredLayers[1]])
    else:
        x = convBlock(x, 64, transferedLayer)

    if (transferredLayers != [] and (transferedLayer == 2 or transferedLayer == 3)):
        x = convBlock(x, 128, transferedLayer, [transferredLayers[2], transferredLayers[3]])
    else:
        x = convBlock(x, 128, transferedLayer)

    if (transferredLayers != [] and (transferedLayer == 4 or transferedLayer == 5)):
        x = convBlock(x, 512, transferedLayer, [transferredLayers[4], transferredLayers[5]])
    x = convBlock(x, 512, transferedLayer)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (transferredLayers != [] and transferedLayer == 6):
        x = QuantaLayer('elu')([transferredLayers[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)
    
    return x

