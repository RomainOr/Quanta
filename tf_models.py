import tensorflow as tf
from sys import exit
from binaryChoiceActivation import binaryChoiceActivation

v = 2.95
initScalarValue = [v/2, -v/2] #softmax(2.94, 0) ~ (0.95, 0.05) softmax(4.59, 0) ~ (0.99, 0.01)

eta               = 2e-4	#Learning rate   
etaTF 			  = 1e-1	#Learning rate for transferability factors 
etaDecay	 	  = 1e-6	#lr decay for optimizer

def createModel(placeholder, outputSize, withTransfer=None, trsf_type='gradual', trsf_layer=None):
	if trsf_type == 'gradual' and trsf_layer is None: exit()
	if trsf_type == 'gradual':
		return createModel_gradual(placeholder, outputSize, trsf_layer, withTransfer)
	return createModel_coeval(placeholder, outputSize, withTransfer)

####################################################################
## Convblock and template model for GRADUAL eval of transferability
####################################################################

def convBlock_gradual(pInput, size, trsf_layer, transferLayers=[]):
    x = pInput
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)

    if (trsf_layer == 0 or trsf_layer == 2 or trsf_layer == 4):
        if (transferLayers != []):
            # Comment to NOT transfert
            x = binaryChoiceActivation(
                'elu', initScalarValue, etaTF / eta)([transferLayers[0], x])
        else:
            x = tf.keras.layers.Activation('elu')(x)  # Comment to NOT transfert
    else:
        x = tf.keras.layers.Activation('elu')(x)  # Comment to transfert

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (trsf_layer == 1 or trsf_layer == 3 or trsf_layer == 5):
        if (transferLayers != []):
            # Comment to NOT transfert
            x = binaryChoiceActivation(
                'elu', initScalarValue, etaTF / eta)([transferLayers[1], x])
        else:
            x = tf.keras.layers.Activation('elu')(x)  # Comment to NOT transfert
    else:
        x = tf.keras.layers.Activation('elu')(x)  # Comment to transfert
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return x

def createModel_gradual(placeholder, outputSize, trsf_layer, withTransfer=None):
    lay = []
    if (withTransfer != None):
        lay = [withTransfer.layers[i].output for i in range(len(withTransfer.layers)) if type(
            withTransfer.layers[i]).__name__ in ['Conv2D', 'Dense']]
    x = placeholder

    if (lay != [] and (trsf_layer == 0 or trsf_layer == 1)):
        x = convBlock_gradual(x, 64, trsf_layer, [lay[0], lay[1]])
    else:
        x = convBlock_gradual(x, 64, trsf_layer)

    if (lay != [] and (trsf_layer == 2 or trsf_layer == 3)):
        x = convBlock_gradual(x, 128, trsf_layer, [lay[2], lay[3]])
    else:
        x = convBlock_gradual(x, 128, trsf_layer)

    if (lay != [] and (trsf_layer == 4 or trsf_layer == 5)):
        x = convBlock_gradual(x, 512, trsf_layer, [lay[4], lay[5]])
    x = convBlock_gradual(x, 512, trsf_layer)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (lay != [] and trsf_layer == 6):
        x = binaryChoiceActivation(
            'elu', initScalarValue, etaTF / eta)([lay[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)
    return x


####################################################################
## Convblock and template model for co-evaluation of transferability
####################################################################
def convBlock_coeval(pInput, size, transferLayers=[]):
	x = pInput
	x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
	if (transferLayers != []): x = \
		binaryChoiceActivation('elu', initScalarValue, etaTF/eta)([transferLayers[0], x])
	else: x = tf.keras.layers.Activation('elu')(x)
	x = tf.keras.layers.BatchNormalization()(x)

	x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
	if (transferLayers != []): x = \
		binaryChoiceActivation('elu', initScalarValue, etaTF/eta)([transferLayers[1], x])
	else: x = tf.keras.layers.Activation('elu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	
	x = tf.keras.layers.MaxPool2D(2)(x)
	x = tf.keras.layers.Dropout(0.3)(x)	
	return x


def createModel_coeval(placeholder, outputSize, withTransfer=None):
	lay = []
	if (withTransfer != None):
		lay = [withTransfer.layers[i].output     \
		for i in range(len(withTransfer.layers)) \
		if type(withTransfer.layers[i]).__name__ in ['Conv2D', 'Dense']]	
	x = placeholder
	if lay != []: x = convBlock_coeval(x, 64, [lay[0], lay[1]])
	else : x = convBlock_coeval(x, 64)
	
	if lay != []: x = convBlock_coeval(x, 128, [lay[2], lay[3]])
	else : x = convBlock_coeval(x, 128)
	
	if lay != []: x = convBlock_coeval(x, 512, [lay[4], lay[5]])
	else : x = convBlock_coeval(x, 512)

	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(2048)(x)
	if lay != []:  x = binaryChoiceActivation('elu', initScalarValue, etaTF/eta)([lay[6], x])
	else : x = tf.keras.layers.Activation('elu')(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	

	x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)
	return x

