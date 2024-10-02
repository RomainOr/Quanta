import sys
import os


#################################################
############ Tensforflow verbosity ##############
#################################################

# You can also adjust the verbosity by changing the value of TF_CPP_MIN_LOG_LEVEL:
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
# Make sure to put those lines before import tensorflow to be effective.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    #Less verbosity

import random
import numpy as np
import tensorflow as tf


#################################################
##### Set up Determinism or not with QUANTA #####
#################################################

SEED = int(sys.argv[5])
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
# Call the above function with seed value if well defined
if SEED is not None and SEED > 0:
    set_global_determinism(seed=SEED)


#################################################
####### Managing arguments and parameters #######
#################################################
# TODO : Manage exceptions and defaults wrt shell script

# Usage: ./start_expe -o|--outdir=outputdir -r|--repeat=30 -l|--layer=5 -t|--targetTask=cifar10 --seed=0
# Cf. start_expe.sh
outputDir  = sys.argv[1]
currentRun = int(sys.argv[2])
layerToTransfer = int(sys.argv[3])
targetTask = sys.argv[4]

print("Python parameters :")
print("\t Output directory : ", outputDir)
if not os.path.exists(outputDir):
    print('\t\t Creating non-existing output directory')
    os.makedirs(outputDir)
print("\t Current run : ", currentRun)
print("\t Layer : ", layerToTransfer)
print("\t Target task : ", targetTask)
print("\t Seed : ", SEED)
print("\n")

# User defined parameters
trainSource = False
nbOfSamples = 320

learningRate          = 2e-4 
weightDecay           = 1e-6
numberOfEpochsSource  = 1
numberOfEpochsTarget  = 1
batchSizeSource       = 32    
batchSizeTarget       = 32

optimizer   = tf.keras.optimizers.Adam(learningRate, weight_decay=weightDecay)    #Optimizer for gradient descent
loss        = 'categorical_crossentropy'    #Loss giving gradients
metrics     = [tf.keras.metrics.CategoricalAccuracy(), \
               tf.keras.metrics.Precision(),\
               tf.keras.metrics.Recall(),   \
               tf.keras.metrics.FalsePositives(), \
               tf.keras.metrics.FalseNegatives(), \
               tf.keras.metrics.TruePositives(),  \
               tf.keras.metrics.TrueNegatives(),  \
               tf.keras.metrics.TopKCategoricalAccuracy(k=3)]

augmentData          = False   #Flow data augmentation during training
fromPreviousTraining = False   #False for training from scratch, True to start from previous save


#################################################
############## Loading input data ###############
#################################################

print("Loading Data : start")
(trainingSetSource, trainingLabelsSource),(testSetSource, testLabelsSource) = \
    tf.keras.datasets.cifar10.load_data()
if targetTask == 'cifar10':
    (trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
            tf.keras.datasets.cifar10.load_data()
elif targetTask == 'cifar100':
    (trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
            tf.keras.datasets.cifar100.load_data()
else:
    (trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
            tf.keras.datasets.cifar10.load_data()

trainingSetSource = trainingSetSource[:nbOfSamples]
trainingLabelsSource = trainingLabelsSource[:nbOfSamples]
testSetSource = testSetSource[:nbOfSamples]
testLabelsSource = testLabelsSource[:nbOfSamples]

trainingSetTarget = trainingSetTarget[:nbOfSamples]
trainingLabelsTarget = trainingLabelsTarget[:nbOfSamples]
testSetTarget = testSetTarget[:nbOfSamples]
testLabelsTarget = testLabelsTarget[:nbOfSamples]
print("Loading Data : done\n")


#################################################
#### Normalizing data and creating one-hots #####
#################################################

print("Normalizing data and creating one-hots : start")
trainingSetSource    = trainingSetSource/255.
testSetSource    = testSetSource/255.
trainingLabelsSource = tf.keras.utils.to_categorical(trainingLabelsSource, 10)
testLabelsSource = tf.keras.utils.to_categorical(testLabelsSource, 10)

trainingSetTarget = trainingSetTarget/255.
testSetTarget = testSetTarget/255.
if targetTask == 'cifar100':
    trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, 100)
    testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, 100)
else:
    trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, 10)
    testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, 10)
print("Normalizing data and creating one-hots : done\n")


#################################################
######### Building and compiling models #########
#################################################

from models import compileModels

sourceModel, sourceModelCopy, targetModel = compileModels(targetTask, layerToTransfer, optimizer, loss, metrics)


#################################################
######### Model training and evaluation #########
#################################################

from typing import cast
from QuantaLayer import QuantaLayer

# Data augmentation generator
dataAugmentationGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    # https://www.tensorflow.org/guide/keras/preprocessing_layers?hl=fr
    rotation_range=45,              #randomly rotate images in the range (degrees, 0 to 180) - tf.keras.layers.RandomRotation
    width_shift_range=0.1,          #randomly shifting image horizontally - tf.keras.layers.RandomTranslation
    height_shift_range=0.1,         #randomly shifting image vertically - tf.keras.layers.RandomTranslation
    shear_range=0.1,                #set range for random shear
    zoom_range=0.1,                 #set range for random zoom - tf.keras.layers.RandomZoom
    horizontal_flip=True,           #randomly flip images horizontally - tf.keras.layers.RandomFlip
    vertical_flip=True,             #randomly flip images vertially - tf.keras.layers.RandomFlip
)

def train(modelName, dataAugmentation=False, fromPreviousTraining=False): 
    trainingMetrics = {}
    # Callback for exporting accuracy during training
    # modelName : T for target, S for source
    if (modelName == 'T'):
        print("\nTraining target model : start")
        sourceModelCopy.load_weights('./SourceModel_w.h5')
        model = targetModel
        trainingSet    = trainingSetTarget
        trainingLabels = trainingLabelsTarget
        weights = outputDir + "/TargetModel_w.h5"
        noe = numberOfEpochsTarget
        cb  = [tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch,logs: 
                    trainingMetrics.update({epoch : logs})
                )]
        for i in range(len(model.layers)):
            if (type(model.layers[i]).__name__ == "QuantaLayer"):
                cb.append(cast(QuantaLayer, model.layers[i]).getCustomCallback(i))
        bc  = batchSizeTarget
    else :
        print("\nTraining source model : start")
        model = sourceModel
        trainingSet    = trainingSetSource
        trainingLabels = trainingLabelsSource
        weights = "./SourceModel_w.h5"
        noe = numberOfEpochsSource
        cb  = [tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch,logs: 
                    trainingMetrics.update({epoch : logs})
                )]
        bc  = batchSizeSource

    if dataAugmentation: 
        dataAugmentationGenerator.fit(trainingSet, augment=True)
        if fromPreviousTraining: model.load_weights(weights)
        model.fit(
                dataAugmentationGenerator.flow(
                    trainingSet, trainingLabels, batch_size=bc), epochs=noe, callbacks=cb)
    else: 
        if fromPreviousTraining: model.load_weights(weights)
        model.fit(trainingSet, trainingLabels, epochs=noe, callbacks=cb, batch_size=bc)
    print("Training model : done\n")

    print("Saving model parameters : start")
    model.save_weights(weights)
    print("Saving model parameters : done\n")
    return trainingMetrics

def test(modelName): 
    if (modelName == 'T'):
        print("Testing target model : start")
        sourceModelCopy.load_weights("./SourceModel_w.h5")
        model = targetModel
        testSet    = testSetTarget
        testLabels = testLabelsTarget
        weights = outputDir + "/TargetModel_w.h5"
    else :
        print("Testing source model : start")
        model = sourceModel
        testSet    = testSetSource
        testLabels = testLabelsSource
        weights = "./SourceModel_w.h5"

    model.load_weights(weights)
    metrics = model.evaluate(testSet, testLabels, return_dict=True)
    print("Testing model : done\n")
    return metrics

trainingMetricsOfTarget = train('T', augmentData, fromPreviousTraining)
testingMetricsOfSource = test('S')
testingMetricsOfTarget = test('T')


#################################################
############### Exporting metrics ###############
#################################################

s = 'model_run_'+str(currentRun)+'_layer_'+str(layerToTransfer)

f = open(outputDir+'/training_metrics_of_' + targetModel._name + '_' + s +'.jsonl', 'a')
f.write(str(trainingMetricsOfTarget)+'\n')
f.close()

f = open(outputDir+'/testing_metrics_of_' + sourceModel._name + '_' + s +'.jsonl', 'a')
f.write(str(testingMetricsOfSource)+'\n')
f.close()

f = open(outputDir+'/testing_metrics_of_' + targetModel._name + '_' + s +'.jsonl', 'a')
f.write(str(testingMetricsOfTarget)+'\n')
f.close()

sourceModel.save(outputDir + '/' + sourceModel._name + '_' + s +'.keras')
targetModel.save(outputDir + '/' + targetModel._name + '_' + s +'.keras')

print("Final testing categorical accuracy of source :" +str(testingMetricsOfSource['categorical_accuracy']))
print("Final testing categorical accuracy of target :" +str(testingMetricsOfTarget['categorical_accuracy']))
