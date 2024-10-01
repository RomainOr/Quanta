import tensorflow as tf

class QuantaCustomCallback(tf.keras.callbacks.Callback):

        def __init__(self):
            super().__init__()
            self._quantaWeights = []
            self._quantas = []

        def _monitorQuantaLayers(self, model):
            s= ""
            for l in range(len(model.layers)):
                if (type(model.layers[l]).__name__ == "QuantaLayer"):
                    # Collect information to display
                    quantaWeights = model.layers[l].get_weights()[0][0]
                    s += '\t Quanta weights : ' + str(quantaWeights) + '\n'
                    quantas = tf.nn.softmax(quantaWeights)
                    s += '\t Quanta value of Source : ' + str(quantas.numpy()[0]) + '\n'
                    s += '\t Quanta value of Target : ' + str(quantas.numpy()[1]) + '\n'
                    # Store computed wieghts of quanta and value of source quanta
                    self._quantaWeights.append(quantaWeights)
                    self._quantas.append(quantas)
            return s

        def getQuantaWeights(self):
            return self._quantaWeights

        def getQuantas(self):
            return self._quantas

        def on_epoch_begin(self, epoch, logs=None):
            print('\nQuanta layer :\n', self._monitorQuantaLayers(self.model))

        def on_epoch_end(self, epoch, logs=None):
            print('\nQuanta layer :\n', self._monitorQuantaLayers(self.model))