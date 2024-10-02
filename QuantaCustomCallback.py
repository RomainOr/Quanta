import tensorflow as tf

class QuantaCustomCallback(tf.keras.callbacks.Callback):

        def __init__(self):
            super().__init__()
            self._quantaWeights = []
            self._quantas = []
            self._nbLayer = -1

        def _monitorQuantaLayer(self, layer):
            s= '\n\t Quanta layer ' + str(self._nbLayer) + ' :\n'
            # Collect information to display
            quantaWeights = layer.get_weights()[0][0]
            s += '\t\t Quanta weights : ' + str(quantaWeights) + '\n'
            quantas = tf.nn.softmax(quantaWeights)
            s += '\t\t Quanta value of Source : ' + str(quantas.numpy()[0]) + '\n'
            s += '\t\t Quanta value of Target : ' + str(quantas.numpy()[1]) + '\n'
            # Store computed wieghts of quanta and value of source quanta
            self._quantaWeights.append(quantaWeights)
            self._quantas.append(quantas)
            return s

        def getQuantaWeights(self):
            return self._quantaWeights

        def getQuantas(self):
            return self._quantas
        
        def setNbrLayer(self, nbLayer):
            if self._nbLayer == -1 :
                self._nbLayer = nbLayer

        def on_train_begin(self, logs=None):
            print(self._monitorQuantaLayer(self.model.layers[self._nbLayer]))

        def on_epoch_end(self, epoch, logs=None):
            print(self._monitorQuantaLayer(self.model.layers[self._nbLayer]))