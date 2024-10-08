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

optimizer   = tf.keras.optimizers.Adam(learningRate, weight_decay=weightDecay)
loss        = tf.keras.losses.CategoricalCrossentropy()
metrics     = [tf.keras.metrics.CategoricalAccuracy(), 
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall(),   
               tf.keras.metrics.FalsePositives(),
               tf.keras.metrics.FalseNegatives(),
               tf.keras.metrics.TruePositives(),
               tf.keras.metrics.TrueNegatives(),
               tf.keras.metrics.TopKCategoricalAccuracy(k=3)]

augmentData          = False   #Flow data augmentation during training
fromPreviousTraining = False   #False for training from scratch, True to start from previous save


#################################################
############## Loading input data ###############
#################################################

print("Loading Data : start")
(trainingSetSource, trainingLabelsSource),(testSetSource, testLabelsSource) = \
    tf.keras.datasets.cifar10.load_data()
nb_of_source_classes = 10

nb_of_target_classes = -1
if targetTask == 'cifar100':
    (trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
            tf.keras.datasets.cifar100.load_data()
    nb_of_target_classes = 100
else:
    (trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
            tf.keras.datasets.cifar10.load_data()
    nb_of_target_classes = 10
input_shape = trainingSetTarget.shape[1:]

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
############### Creating one-hots ###############
#################################################

print("Creating one-hots : start")
trainingLabelsSource = tf.keras.utils.to_categorical(trainingLabelsSource, nb_of_source_classes)
testLabelsSource = tf.keras.utils.to_categorical(testLabelsSource, nb_of_source_classes)

trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, nb_of_target_classes)
testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, nb_of_target_classes)
print("Creating one-hots : done\n")


#################################################
######### Building and compiling models #########
#################################################

from models import compileModels

sourceModel, sourceModelCopy, targetModel = compileModels(input_shape, targetTask, layerToTransfer, optimizer, loss, metrics, augmentData)


#################################################
######### Model training and evaluation #########
#################################################

from typing import cast
from QuantaLayer import QuantaLayer

def train(modelName, fromPreviousTraining=False): 
    trainingMetrics = {}
    # Callback for exporting accuracy during training
    # modelName : T for target, S for source
    if (modelName == 'T'):
        print("\nTraining target model : start")
        sourceModelCopy.load_weights('./SourceModel.weights.h5')
        model = targetModel
        trainingSet    = trainingSetTarget
        trainingLabels = trainingLabelsTarget
        weights = outputDir + "/TargetModel.weights.h5"
        noe = numberOfEpochsTarget
        cb  = [tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch,logs: 
                    trainingMetrics.update({epoch : logs})
                )]
        for i in range(len(model.layers)):
            if (type(model.layers[i]).__name__ == "QuantaLayer"):
                cb.append(cast(QuantaLayer, model.layers[i]).getCustomCallback(i))
    else :
        print("\nTraining source model : start")
        model = sourceModel
        trainingSet    = trainingSetSource
        trainingLabels = trainingLabelsSource
        weights = "./SourceModel.weights.h5"
        noe = numberOfEpochsSource
        cb  = [tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch,logs: 
                    trainingMetrics.update({epoch : logs})
                )]
    
    if fromPreviousTraining: 
        model.load_weights(weights)

    train_dataset = tf.data.Dataset.from_tensor_slices((trainingSet, trainingLabels))
    train_dataset = train_dataset.batch(32).map(lambda x, y: (x, y))

    model.fit(train_dataset, epochs=noe, callbacks=cb)
    print("Training model : done\n")

    print("Saving model parameters : start")
    model.save_weights(weights)
    print("Saving model parameters : done\n")
    return trainingMetrics

def test(modelName): 
    if (modelName == 'T'):
        print("Testing target model : start")
        sourceModelCopy.load_weights("./SourceModel.weights.h5")
        model = targetModel
        testSet    = testSetTarget
        testLabels = testLabelsTarget
        weights = outputDir + "/TargetModel.weights.h5"
    else :
        print("Testing source model : start")
        model = sourceModel
        testSet    = testSetSource
        testLabels = testLabelsSource
        weights = "./SourceModel.weights.h5"

    model.load_weights(weights)

    test_dataset = tf.data.Dataset.from_tensor_slices((testSet, testLabels))
    test_dataset = test_dataset.batch(32).map(lambda x, y: (x, y))
    
    metrics = model.evaluate(test_dataset, return_dict=True)
    print("Testing model : done\n")
    return metrics

testingMetricsOfSource = test('S')

# Important to reset states of each used metric as they are shared by both models
for metric in metrics :
    cast(tf.keras.metrics.Metric, metric).reset_state()
# Potentially, be also carefull about that point :
# https://stackoverflow.com/questions/65923011/keras-tensoflow-full-reset

trainingMetricsOfTarget = train('T', fromPreviousTraining)
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
