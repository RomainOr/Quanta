import tensorflow as tf

class QuantaCustomCallback(tf.keras.callbacks.Callback):

        def __init__(self):
            super().__init__()
            self._quanta_weights = []
            self._quantas = []
            self._position_of_transfered_layer = -1

        def _monitorQuantaLayer(self, layer):
            s= '\n\t Quanta layer ' + str(self._position_of_transfered_layer) + ' :\n'
            # Collect information to display
            quanta_weights = layer.get_weights()[0][0]
            s += '\t\t Quanta weights : ' + str(quanta_weights) + '\n'
            quantas = tf.nn.softmax(quanta_weights)
            s += '\t\t Quanta value of Source : ' + str(quantas.numpy()[0]) + '\n'
            s += '\t\t Quanta value of Target : ' + str(quantas.numpy()[1]) + '\n'
            # Store computed wieghts of quanta and value of source quanta
            self._quanta_weights.append(quanta_weights)
            self._quantas.append(quantas)
            return s

        def get_quanta_weights(self):
            return self._quanta_weights

        def get_quantas(self):
            return self._quantas
        
        def setNbrLayer(self, position_of_transfered_layer):
            if self._position_of_transfered_layer == -1 :
                self._position_of_transfered_layer = position_of_transfered_layer

        def on_train_begin(self, logs=None):
            print(self._monitorQuantaLayer(self.model.layers[self._position_of_transfered_layer]))

        def on_epoch_end(self, epoch, logs=None):
            print(self._monitorQuantaLayer(self.model.layers[self._position_of_transfered_layer]))