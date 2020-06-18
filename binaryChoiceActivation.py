from tensorflow.python.keras import activations
from tensorflow.python.keras.initializers import constant
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops.math_ops import scalar_mul, add
from tensorflow.python.ops.array_ops import split
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow.python.ops.custom_gradient import custom_gradient
from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.framework.constant_op import constant as c
from tensorflow import float32

from numpy import exp

import tensorflow as tf

L = 6 # total number of layers 0..6
modifiers = [exp(-l)*1000. for l in range(0,7)]

class binaryChoiceActivation(_Merge):
	def __init__(self, activation, initScalarWeightValue=[0.0, 0.0],
				 learningRateModifier=1000.0, layer_num=None, **kwargs):
		self.supports_masking = True
		self.activation = activations.get(activation)
		self.initScalarWeightValue = initScalarWeightValue
		# lrm: gradient boost for the lambdas
		self.lrm = learningRateModifier if layer_num is None \
							else ( (1./1.) * learningRateModifier ) # condition to use modifiers
		print('Learning rate modifier :', self.lrm)
		super(binaryChoiceActivation, self).__init__(**kwargs)

	@tf_utils.shape_type_conversion
	def build(self, input_shape):
		self.kernel = self.add_weight(name='scalarWeights', shape=[1, 2], dtype=float32, \
								initializer=constant(self.initScalarWeightValue), trainable=True)
		super(binaryChoiceActivation, self).build(input_shape)

	def getGradientPumpFunction(self):	
		@custom_gradient
		def pumpLambdaGradient(x):
			def grad(dy):
				scal = c(self.lrm)
				return scalar_mul(scal, dy)
			return x, grad
		return pumpLambdaGradient

	#############
	# Merge the outputs of transferred source layer l and target layer l
	def _merge_function(self, inputs):  
		"""
		Version 1 : tanh/sigmoid
		Version 2 : softmax
		"""
		lambdaScalar = split(self.getGradientPumpFunction()(activations.get('sigmoid')(self.kernel)), 2, -1)
		lambdaScalar = [reshape(lambdaScalar[i], []) for i in range(len(lambdaScalar))]
		# TF v.2: A_T = f(Z_S*Lambda_S + Z_T*lambda_T)
		#return self.activation(add(scalar_mul(lambdaScalar[0], inputs[0]), 
		#	scalar_mul(lambdaScalar[1], inputs[1])))
		#####
		# TF v.1 (first concept from Amarin
		# A_T = f(Z_S*Lambda_S + (1-Lambda_S)*Z_T)
		return self.activation(add(scalar_mul(lambdaScalar[0], inputs[0]),
			scalar_mul( (1-lambdaScalar[0]), inputs[1] )))

