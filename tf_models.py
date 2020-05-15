import tensorflow as tf
from sys import exit
from binaryChoiceActivation import binaryChoiceActivation

v = 0.02 # v1: 3.5    v2: 2.95
initScalarValue = [v/2, -v/2] #softmax(1.475, -1.475) ~ (0.95, 0.049)

etaTF 			  = 1e-5 # v1 boost 5e-2 # v2 1e-1	#Learning rate for transferability factors 

eta               = 2e-4 # v1 boost 1e-3 # v2 2e-4	#Learning rate   
etaDecay	 	  = 1e-6	#lr decay for optimizer

def learning_rates(): return eta, etaTF

def createModel(placeholder, outputSize, sourceModel=None, trsf_type='gradual', trsf_layer=None):
	if trsf_type == 'gradual' and trsf_layer is None: exit()
	if trsf_type == 'gradual':
		return createModel_gradual(placeholder, outputSize, trsf_layer, sourceModel)
	return createModel_coeval(placeholder, outputSize, sourceModel)

####################################################################
## Convblock and template model for GRADUAL eval of transferability
####################################################################

def convBlock_gradual(pInput, size, trsf_layer, transferredLayers=[]):
    x = pInput
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)

    if (trsf_layer == 0 or trsf_layer == 2 or trsf_layer == 4):
        if (transferredLayers != []):
            # Comment to NOT transfert
            x = binaryChoiceActivation(
                'elu', initScalarValue, etaTF / eta)([transferredLayers[0], x])
        else:
            x = tf.keras.layers.Activation('elu')(x)  # Comment to NOT transfert
    else:
        x = tf.keras.layers.Activation('elu')(x)  # Comment to transfert

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (trsf_layer == 1 or trsf_layer == 3 or trsf_layer == 5):
        if (transferredLayers != []):
            # Comment to NOT transfert
            x = binaryChoiceActivation(
                'elu', initScalarValue, etaTF / eta)([transferredLayers[1], x])
        else:
            x = tf.keras.layers.Activation('elu')(x)  # Comment to NOT transfert
    else:
        x = tf.keras.layers.Activation('elu')(x)  # Comment to transfert
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return x

def createModel_gradual(placeholder, outputSize, trsf_layer, sourceModel=None):
    transferredLayers = []
    if (sourceModel != None):
        transferredLayers = [sourceModel.layers[i].output for i in range(len(sourceModel.layers)) if type(
            sourceModel.layers[i]).__name__ in ['Conv2D', 'Dense']]
    x = placeholder

    if (transferredLayers != [] and (trsf_layer == 0 or trsf_layer == 1)):
        x = convBlock_gradual(x, 64, trsf_layer, [transferredLayers[0], transferredLayers[1]])
    else:
        x = convBlock_gradual(x, 64, trsf_layer)

    if (transferredLayers != [] and (trsf_layer == 2 or trsf_layer == 3)):
        x = convBlock_gradual(x, 128, trsf_layer, [transferredLayers[2], transferredLayers[3]])
    else:
        x = convBlock_gradual(x, 128, trsf_layer)

    if (transferredLayers != [] and (trsf_layer == 4 or trsf_layer == 5)):
        x = convBlock_gradual(x, 512, trsf_layer, [transferredLayers[4], transferredLayers[5]])
    x = convBlock_gradual(x, 512, trsf_layer)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (transferredLayers != [] and trsf_layer == 6):
        x = binaryChoiceActivation(
            'elu', initScalarValue, etaTF / eta)([transferredLayers[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)
    return x


####################################################################
## Convblock and template model for co-evaluation of transferability
####################################################################
def convBlock_coeval(pInput, size, transferredLayers=[], layers_num=None):
	x = pInput
	x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
	if (transferredLayers != [] and layers_num is not None): x = \
		binaryChoiceActivation('elu', initScalarValue, etaTF/eta, layers_num[0])([transferredLayers[0], x])
	else: x = tf.keras.layers.Activation('elu')(x)
	x = tf.keras.layers.BatchNormalization()(x)

	x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
	if (transferredLayers != [] and layers_num is not None): x = \
		binaryChoiceActivation('elu', initScalarValue, etaTF/eta, layers_num[1])([transferredLayers[1], x])
	else: x = tf.keras.layers.Activation('elu')(x)
	x = tf.keras.layers.BatchNormalization()(x)
	
	x = tf.keras.layers.MaxPool2D(2)(x)
	x = tf.keras.layers.Dropout(0.3)(x)	
	return x


def createModel_coeval(placeholder, outputSize, sourceModel=None):
	transferredLayers = []
	if (sourceModel != None):
		transferredLayers = [sourceModel.layers[i].output     \
		for i in range(len(sourceModel.layers)) \
		if type(sourceModel.layers[i]).__name__ in ['Conv2D', 'Dense']]	
	
	x = placeholder

	if transferredLayers != []: 
		x = convBlock_coeval(x, 64, [transferredLayers[0], transferredLayers[1]], layers_num=(0,1))
	else : x = convBlock_coeval(x, 64)
	
	if transferredLayers != []:
		x = convBlock_coeval(x, 128, [transferredLayers[2], transferredLayers[3]], layers_num=(2,3))
	else : x = convBlock_coeval(x, 128)
	
	if transferredLayers != []:
		x = convBlock_coeval(x, 512, [transferredLayers[4], transferredLayers[5]], layers_num=(4,5))
	else : x = convBlock_coeval(x, 512)

	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(2048)(x)
	if transferredLayers != []:
		x = binaryChoiceActivation('elu', initScalarValue, etaTF/eta, layer_num=6)\
								  ([transferredLayers[6], x])
	else : x = tf.keras.layers.Activation('elu')(x)
	x = tf.keras.layers.Dropout(0.5)(x)

	x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)

	return x

