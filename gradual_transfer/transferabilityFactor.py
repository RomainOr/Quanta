import os
import math as m
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from binaryChoiceActivation import binaryChoiceActivation

import sys

print(tf.__version__)
print(tf.keras.__version__)

tf.compat.v1.disable_eager_execution()

# This script must be called with a wrapper script
# The arguments are:
#   argv[0] the current layer that is tested
#   argv[1] the current experiment (30 experiment should be done)
#   argv[2] the output directory to export the results

# Gradual Transfer Learning parameters
layer_transfrd = int(sys.argv[1])  # test transfer up to layer_transfrd
current_expe = int(sys.argv[2])  # current experiment (out of 30) [0;30[

print((layer_transfrd))
print((current_expe))

print("[transferabilityFactor.py]   Current layer: %d / Current run: %d" %
      (layer_transfrd, current_expe))
out_dirname = sys.argv[3]

# Parametres
eta = 2e-4  # Learning rate
etaTF = 1e-1  # Learning rate for transferability factors
etaDecay = 1e-6  # lr decay for optimizer
numberOfEpochsSource = 60  # number of time model is trained over dataset
numberOfEpochsWitness = 60  # number of time model is trained over dataset
numberOfEpochsTarget = 60  # number of time model is trained over dataset
batchSizeSource = 32
batchSizeTarget = 32

optimizer = tf.keras.optimizers.Adam(
    eta, decay=etaDecay)  # Optimizer for gradient descent
loss = 'categorical_crossentropy'  # Loss giving gradients
accuracy = ['accuracy']  # Metric for accuracy

augmentData = True  # Flow  data augmentation during training
# False for training from scratch, True to start from previous save
fromPreviousTraining = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Less verbosity

v = 2.95
# softmax(2.94, 0) ~ (0.95, 0.05) softmax(4.59, 0) ~ (0.99, 0.01)
initScalarValue = [v / 2, -v / 2]


# Datasets
(trainingSetSource, trainingLabelsSource), (testSetSource,
                                            testLabelsSource) = tf.keras.datasets.cifar10.load_data()
(trainingSetTarget, trainingLabelsTarget), (testSetTarget,
                                            testLabelsTarget) = tf.keras.datasets.cifar10.load_data()

trainingSetSource    = trainingSetSource / 255.
trainingLabelsSource = tf.keras.utils.to_categorical(trainingLabelsSource, 10)
testSetSource    = testSetSource / 255.
testLabelsSource = tf.keras.utils.to_categorical(testLabelsSource, 10)

trainingSetTarget    = trainingSetTarget / 255.
trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, 10)
testSetTarget    = testSetTarget / 255.
testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, 10)

inputPlaceholderSource = tf.keras.Input([32, 32, 3], name='inputHolderSource')
inputPlaceholderTarget = tf.keras.Input([32, 32, 3], name='inputHolderTarget')


def convBlock(pInput, size, transferLayers=[]):
    x = pInput
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)

    if (layer_transfrd == 0 or layer_transfrd == 2 or layer_transfrd == 4):
        if (transferLayers != []):
            # Comment to NOT transfert
            x = binaryChoiceActivation(
                'elu', initScalarValue, etaTF / eta)([transferLayers[0], x])
        else:
            x = tf.keras.layers.Activation('elu')(
                x)  # Comment to NOT transfert
    else:
        x = tf.keras.layers.Activation('elu')(x)  # Comment to transfert

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (layer_transfrd == 1 or layer_transfrd == 3 or layer_transfrd == 5):
        if (transferLayers != []):
            # Comment to NOT transfert
            x = binaryChoiceActivation(
                'elu', initScalarValue, etaTF / eta)([transferLayers[1], x])
        else:
            x = tf.keras.layers.Activation('elu')(
                x)  # Comment to NOT transfert
    else:
        x = tf.keras.layers.Activation('elu')(x)  # Comment to transfert
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return x


# Model Template
def createModel(placeholder, outputSize, withTransfer=None):
    lay = []
    if (withTransfer != None):
        lay = [withTransfer.layers[i].output for i in range(len(withTransfer.layers)) if type(
            withTransfer.layers[i]).__name__ in ['Conv2D', 'Dense']]
    x = placeholder

    if (lay != [] and (layer_transfrd == 0 or layer_transfrd == 1)):
        x = convBlock(x, 64, [lay[0], lay[1]])
    else:
        x = convBlock(x, 64)

    if (lay != [] and (layer_transfrd == 2 or layer_transfrd == 3)):
        x = convBlock(x, 128, [lay[2], lay[3]])
    else:
        x = convBlock(x, 128)

    if (lay != [] and (layer_transfrd == 4 or layer_transfrd == 5)):
        x = convBlock(x, 512, [lay[4], lay[5]])
    x = convBlock(x, 512)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (lay != [] and layer_transfrd == 6):
        x = binaryChoiceActivation(
            'elu', initScalarValue, etaTF / eta)([lay[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(outputSize, activation='softmax')(x)
    return x

# Data augmentation tensor
trainGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=45,
    width_shift_range=0.1,  # randomly shifting image horizontally
    height_shift_range=0.1,  # randomly shifting image vertically
    shear_range=0.1,  # set range for random shear
    zoom_range=0.1,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    fill_mode='nearest',  # fillmode for image manipulation
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=True,  # randomly flip images vertially
    rescale=None,
    preprocessing_function=None,
    data_format=None)


# Building models
print("building  models...")
NNSource = tf.keras.Model(inputs=inputPlaceholderSource,
                          outputs=createModel(inputPlaceholderSource, 10))
NNSourceCopy = tf.keras.Model(
    inputs=inputPlaceholderTarget, outputs=createModel(inputPlaceholderTarget, 10))
NNTarget = tf.keras.Model(inputs=inputPlaceholderTarget, outputs=createModel(
    inputPlaceholderTarget, 10, withTransfer=NNSourceCopy))
NNWitness = tf.keras.Model(inputs=inputPlaceholderTarget,
                           outputs=createModel(inputPlaceholderTarget, 10))

for l in NNSourceCopy.layers:
    l.trainable = False
NNSource.compile(optimizer, loss, accuracy)
NNTarget.compile(optimizer, loss, accuracy)
NNWitness.compile(optimizer, loss, accuracy)

NotLoadSource = True
NotLoadTarget = True
NotLoadWitness = True

nbrLambdas = len([NNTarget.layers[i].get_weights() for i in range(len(
    NNTarget.layers)) if (type(NNTarget.layers[i]).__name__ == "binaryChoiceActivation")])

# callbacks for retrieving lambdas
lambdas = [[] for i in range(0, nbrLambdas)]


def lambdasMonitoring(b, w):
    if (b % 10 == 0 and b is not 0):
        weights = [np.array(w[i])[0][0] for i in range(len(w))]
        for i in range(0, nbrLambdas):
            lambdas[i] += [m.exp(weights[i][0]) /
                           (m.exp(weights[i][0]) + m.exp(weights[i][1]))]

scalarWeightCallback = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: lambdasMonitoring(batch, [NNTarget.layers[
                                                         i].get_weights() for i in range(len(NNTarget.layers)) if (type(NNTarget.layers[i]).__name__ == "binaryChoiceActivation")]))

printLambdas = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(
    "  TF : " + str(lambdas[0][-1]) + "..." + str(lambdas[-1][-1])))


# Training and evaluating models
# modelName : T for target, S for source, W for witness
def train(modelName, dataAugmentation=False, fromPreviousTraining=False):
    if (modelName == 'T'):
        print("training target model...")
        NNSourceCopy.load_weights("NNSource_w.h5")
        model = NNTarget
        trainingSet = trainingSetTarget
        trainingLabels = trainingLabelsTarget
        weights = "NNTarget_w.h5"
        noe = numberOfEpochsTarget
        cb = [scalarWeightCallback, printLambdas]
        bc = batchSizeTarget
        NotLoadTarget = False

    elif (modelName == 'S'):
        print("training source model...")
        model = NNSource
        trainingSet = trainingSetSource
        trainingLabels = trainingLabelsSource
        weights = "NNSource_w.h5"
        cb = []
        noe = numberOfEpochsSource
        bc = batchSizeSource
        NotLoadSource = False
    else:
        print("training witness model...")
        model = NNWitness
        trainingSet = trainingSetTarget
        trainingLabels = trainingLabelsTarget
        weights = "NNWitness_w.h5"
        cb = []
        noe = numberOfEpochsWitness
        bc = batchSizeSource
        NotLoadWitness = False

    if dataAugmentation:
        trainGenerator.fit(trainingSet, augment=True)
        if fromPreviousTraining:
            model.load_weights(weights)
        model.fit_generator(trainGenerator.flow(
            trainingSet, trainingLabels, batch_size=bc), epochs=noe, callbacks=cb, )
    else:
        if fromPreviousTraining:
            model.load_weights(weights)
        NNSource.fit(trainingSet, trainingLabels,
                     epochs=noe, callbacks=cb, batch_size=bc)
    print("saving weights...")
    model.save_weights(weights)


def test(modelName):  # testing
    if (modelName == 'T'):
        print("testing target model...")
        NNSourceCopy.load_weights("NNSource_w.h5")
        model = NNTarget
        testSet = testSetTarget
        testLabels = testLabelsTarget
        weights = "NNTarget_w.h5"
        snl = NotLoadTarget

    elif (modelName == 'S'):
        print("testing source model...")
        model = NNSource
        testSet = testSetSource
        testLabels = testLabelsSource
        weights = "NNSource_w.h5"
        snl = NotLoadWitness
    else:
        print("testing witness model...")
        model = NNWitness
        testSet = testSetTarget
        testLabels = testLabelsTarget
        weights = "NNWitness_w.h5"
        snl = NotLoadSource

    if snl:
        model.load_weights(weights)
    _, accuracy = model.evaluate(testSet, testLabels)

    f = open(out_dirname + str(layer_transfrd) +
             '/acc', 'w')  # export test accuracy
    f.write(str(accuracy) + '\n')
    f.close()

    print("accuracy :" + str(accuracy) + "\n")


def writeFactors(l, e):
    f = open(out_dirname + str(layer_transfrd) + '/' + str(e) + '.txt', 'w')
    out = ""
    for i in range(0, nbrLambdas):
        for j in range(0, len(l[i])):
            out += str(l[i][j]) + '\n'
        out += "--SPLIT--"
    f.write(out)
    f.close()


def plotLambdaTraining():
    l = []
    print(int(len(trainingSetTarget) / batchSizeTarget / 10) * numberOfEpochsTarget)
    for j in range(0, nbrLambdas):
        l += [[]]
        for k in range(0, int(len(trainingSetTarget) / batchSizeTarget / 10) * numberOfEpochsTarget):
            l[j] += [[]]
            for i in range(0, 30):
                l[j][k] += [0.0]

    for i in range(0, 30):
        f = open(out_dirname + layer_transfrd + '/' + str(i) + '.txt', 'r')
        a = f.read().split(',')
        for j in range(0, nbrLambdas):
            k = 0
            for TF in a[:-1][j].split():
                l[j][k][i] = float(TF)
                k += 1

    f = open(out_dirname + layer_transfrd + '/text' + str(k) + '.txt', 'w')
    for i in range(0, nbrLambdas):
        print(np.mean(l[i][-1]))
        print(np.std(l[i][-1], -1))
        for j in range(0, len(l[i])):
            f.write(str(np.mean(l[i][j])))
            f.write(' ')
        f.write(',')
    f.close()


def main1():
    with tf.compat.v1.Session() as s:
        s.run(tf.compat.v1.global_variables_initializer())
        s.run(tf.compat.v1.local_variables_initializer())

        # NNTarget.summary()

        #train('S', augmentData, fromPreviousTraining)
        # test('S') #0.8358

        #train('W', augmentData, fromPreviousTraining)
        # test('W') #0.5628

        # transfer TF
        if not os.path.exists(out_dirname + str(layer_transfrd) + '/'):
            os.makedirs(out_dirname + str(layer_transfrd) + '/')
        for j in range(0, nbrLambdas):
            lambdas[j] = []
        print("\nExperiment number  " + str(current_expe))
        train('T', augmentData, fromPreviousTraining)
        writeFactors(lambdas, current_expe)
        test('T')  # test target model
        print("Cleaning up...\n")

        # plotLambdaTraining()

if __name__ == "__main__":
    main1()
