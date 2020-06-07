import tensorflow as tf
from sys import exit
from quanta_layer import quanta_layer

v = 2.2 # v = 2.2 => lbd ~0.90
initScalarValue = [v/2, -v/2] #softmax(1.1, -1.1) ~ (0.90, 0.10)

lrm      = 1.0  # learning rate modifier (testing) 
eta      = 2e-4 # learning rate   
etaDecay = 1e-6 # weight decay

def get_vinit(): return v
def learning_rates(): return eta,lrm

####################################################################
## Convblock definition
####################################################################
def convBlock(pInput, size, trsf_layer, transferredLayers=[]):
    x = pInput
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)

    if (trsf_layer == 0 or trsf_layer == 2 or trsf_layer == 4):
        if (transferredLayers != []):
            x = quanta_layer(
                'elu', initScalarValue, lrm)([transferredLayers[0], x])
        else:
            x = tf.keras.layers.Activation('elu')(x)
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (trsf_layer == 1 or trsf_layer == 3 or trsf_layer == 5):
        if (transferredLayers != []):
            x = quanta_layer(
                'elu', initScalarValue, lrm)([transferredLayers[1], x])
        else:
            x = tf.keras.layers.Activation('elu')(x)
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return x

####################################################################
## Create full model using convblocks
####################################################################
def createModel(placeholder, outputSize, trsf_layer, sourceModel=None):
    transferredLayers = []
    if (sourceModel != None):
        transferredLayers = [sourceModel.layers[i].output \
                for i in range(len(sourceModel.layers))   \
                if type(sourceModel.layers[i]).__name__ in ['Conv2D', 'Dense']]
    x = placeholder

    if (transferredLayers != [] and (trsf_layer == 0 or trsf_layer == 1)):
        x = convBlock(x, 64, trsf_layer, [transferredLayers[0], transferredLayers[1]])
    else:
        x = convBlock(x, 64, trsf_layer)

    if (transferredLayers != [] and (trsf_layer == 2 or trsf_layer == 3)):
        x = convBlock(x, 128, trsf_layer, [transferredLayers[2], transferredLayers[3]])
    else:
        x = convBlock(x, 128, trsf_layer)

    if (transferredLayers != [] and (trsf_layer == 4 or trsf_layer == 5)):
        x = convBlock(x, 512, trsf_layer, [transferredLayers[4], transferredLayers[5]])
    x = convBlock(x, 512, trsf_layer)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (transferredLayers != [] and trsf_layer == 6):
        x = quanta_layer(
            'elu', initScalarValue, lrm)([transferredLayers[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)
    return x

