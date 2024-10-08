"""Module of the quanta custom callback class."""

from tensorflow.keras.callbacks import Callback
from tensorflow.nn import softmax


class QuantaCustomCallback(Callback):
    """Class related to quantas that enable to track the evolution
    of the quantas aand their related weights."""

    def __init__(self):
        super().__init__()
        self._quanta_weights = []
        self._quantas = []
        self._position_of_transfered_layer = -1

    def _monitor_quanta_layer(self, layer):
        s = '\n\t Quanta layer ' + \
            str(self._position_of_transfered_layer) + ' :\n'
        # Collect information to display
        quanta_weights = layer.get_weights()[0][0]
        s += '\t\t Quanta weights : ' + str(quanta_weights) + '\n'
        quantas = softmax(quanta_weights)
        s += '\t\t Quanta value of Source : ' + str(quantas.numpy()[0]) + '\n'
        s += '\t\t Quanta value of Target : ' + str(quantas.numpy()[1]) + '\n'
        # Store computed wieghts of quanta and value of source quanta
        self._quanta_weights.append(quanta_weights)
        self._quantas.append(quantas)
        return s

    def get_quanta_weights(self):
        """Return the quanta weight values tracked in the custom callback."""
        return self._quanta_weights

    def get_quantas(self):
        """Return the quanta values tracked in the custom callback."""
        return self._quantas

    def set_position_of_transfered_layer(self, position_of_transfered_layer):
        """Setter that enable to link a quanta custom callback to its related layer."""
        if self._position_of_transfered_layer == -1:
            self._position_of_transfered_layer = position_of_transfered_layer

    def on_train_begin(self, logs=None):
        """Function called at the beginning of training."""
        print(self._monitor_quanta_layer(
            self.model.layers[self._position_of_transfered_layer]))

    def on_epoch_end(self, epoch, logs=None):
        """Function called at the end of an epoch."""
        print(self._monitor_quanta_layer(
            self.model.layers[self._position_of_transfered_layer]))
