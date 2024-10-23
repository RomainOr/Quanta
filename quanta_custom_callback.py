"""Module of the quanta custom callback class."""

from tensorflow.keras.callbacks import Callback
from tensorflow.nn import softmax


class QuantaCustomCallback(Callback):
    """
    Class related to quantas that enable to track the evolution
    of the quantas and their related weights. Inherits from Callback.
    """

    def __init__(self):
        super().__init__()
        self._quanta_weights = []
        self._quanta_source = []
        self._quanta_target = []
        self._position_of_transfered_layer = -1

    def _monitor_quanta_layer(self, layer):
        """
        Build the string to display about values of a quanta layer and store 
        computed weights of quanta and value of source quanta within the class.

        Args:
            layer:
                The quanta layer to recover the weights and quanta values.

        Returns:
            A string to display the weights and quanta values if wanted.
        """
        s = '\n\t Quanta layer ' + \
            str(self._position_of_transfered_layer) + ' :\n'
        # Collect information to display
        quanta_weights = layer.get_weights()[0][0]
        s += '\t\t Quanta weights : ' + str(quanta_weights) + '\n'
        quantas = softmax(quanta_weights)
        s += '\t\t Quanta value of Source : ' + str(quantas.numpy()[0]) + '\n'
        s += '\t\t Quanta value of Target : ' + str(quantas.numpy()[1]) + '\n'
        # Store computed weights of quanta and value of source quanta
        self._quanta_weights.append(quanta_weights.tolist())
        self._quanta_source.append(quantas.numpy()[0].item())
        self._quanta_target.append(quantas.numpy()[1].item())
        return s

    def get_quanta_weights(self):
        """
        Return the quanta weight values tracked in the custom callback.

        Returns:
            A list with the quanta weight values of 'source' and 'target'.
        """
        return self._quanta_weights

    def get_quantas(self):
        """
        Return the quanta values tracked in the custom callback.

        Returns:
            A dict with the quanta list values of 'source' and 'target'.
        """
        return {"source": self._quanta_source, "target": self._quanta_target}

    def set_position_of_transfered_layer(self, position_of_transfered_layer):
        """
        Setter that enable to link a quanta custom callback to its related layer.

        Args:
            position_of_transfered_layer:
                The quanta layer to recover the weights and quanta values.
        """
        if self._position_of_transfered_layer == -1:
            self._position_of_transfered_layer = position_of_transfered_layer

    def on_train_begin(self, logs=None):
        """
        Function called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        print(self._monitor_quanta_layer(
            self.model.layers[self._position_of_transfered_layer]))

    def on_epoch_end(self, epoch, logs=None):
        """
        Function called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        print(self._monitor_quanta_layer(
            self.model.layers[self._position_of_transfered_layer]))
