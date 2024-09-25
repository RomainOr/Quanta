from tensorflow.python.keras import activations
from tensorflow.python.keras.initializers import constant
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops.math_ops import scalar_mul, add
from tensorflow.python.ops.array_ops import split
from tensorflow.python.ops.gen_array_ops import reshape
from tensorflow import float32

import tensorflow as tf

class QuantaLayer(_Merge):
    def __init__(self, activation, initLambdaWeights=[1.1, -1.1], **kwargs):
        super(QuantaLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)
        #softmax(1.1, -1.1) ~ (0.90, 0.10)
        self.initLambdaWeights = initLambdaWeights

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.lambdaWeights = self.add_weight(
            name='lambdaWeights', 
            shape=[1, 2], 
            dtype=float32,
            initializer=constant(self.initLambdaWeights),
            trainable=True)
        super(QuantaLayer, self).build(input_shape)
        
    # Merge the outputs of transferred source layer l and target layer l
    def _merge_function(self, inputs):  
        quanta = split((activations.get('softmax')(self.lambdaWeights)), 2, -1)
        quanta = [reshape(quanta[i], []) for i in range(len(quanta))]
        # A_T = f(Z_S*Lambda_S + Z_T*lambda_T)
        return self.activation(
            add(
                scalar_mul(quanta[0], inputs[0]), 
                scalar_mul(quanta[1], inputs[1])))

