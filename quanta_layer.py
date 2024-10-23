"""Module of the quanta layer class."""

from tensorflow.keras import activations
from tensorflow.keras.initializers import constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.ops import split, reshape
from tensorflow.math import add, scalar_mul

from quanta_custom_callback import QuantaCustomCallback


class QuantaLayer(Layer):
    """Class related to quantas between a source model layer and the related
    target model layer."""

    def __init__(self, activation, init_quanta_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        # softmax(1.1, -1.1) ~ (0.90, 0.10)
        if init_quanta_weights is None:
            self._init_quanta_weights = [1.1, -1.1]
        else:
            self._init_quanta_weights = init_quanta_weights
        self._quanta_weights = None
        self._custom_callback = QuantaCustomCallback()
        self._position_of_transfered_layer = -1

    def build(self, input_shape):
        """Create weights that depend on the shape(s) of the input(s)."""
        self._quanta_weights = self.add_weight(
            name='quanta_weights',
            shape=[1, 2],
            initializer=constant(self._init_quanta_weights),
            trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        """Performs the logic of applying the layer to the input arguments.
        A quanta layer applies A_T = f(Z_S \\* Lambda_S + Z_T \\* lambda_T)."""
        quanta = split((activations.get('softmax')
                       (self._quanta_weights)), 2, -1)
        quanta = [reshape(quanta[i], []) for i in range(len(quanta))]
        return self.activation(
            add(
                scalar_mul(quanta[0], inputs[0]),
                scalar_mul(quanta[1], inputs[1])))

    def get_custom_callback(self, position_of_transfered_layer):
        """Return the associated custom callback and set the internal position of transfered 
        layer given in argument."""
        if self._position_of_transfered_layer == -1:
            self._position_of_transfered_layer = position_of_transfered_layer
            self._custom_callback.set_position_of_transfered_layer(
                position_of_transfered_layer)
        return self._custom_callback

    def get_quanta_weights(self):
        """Return the quanta weight values managed by the associated custom callback."""
        return self._custom_callback.get_quanta_weights()

    def get_quantas(self):
        """Return the quanta values managed by the associated custom callback."""
        return self._custom_callback.get_quantas()
