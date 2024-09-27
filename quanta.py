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
# TODO : Manage exceptions and defaults

# Usage: ./start_expe -o|--outdir=outputdir -r|--repeat=30 -l|--layer=5 -t|--targetTask=cifar10 --seed=0
# Cf. start_expe.sh
outputDir  = sys.argv[1]
currentRun = int(sys.argv[2])
trsf_layer = int(sys.argv[3])
targetData = sys.argv[4]

trainSource = False
nbOfSamples = 320

#Parameters
eta                   = 2e-4 # learning rate   
etaDecay              = 1e-6 # weight decay
numberOfEpochsSource  = 1
numberOfEpochsWitness = 1
numberOfEpochsTarget  = 1
batchSizeSource       = 32    
batchSizeTarget       = 32

optimizer   = tf.keras.optimizers.Adam(eta, weight_decay=etaDecay)    #Optimizer for gradient descent
loss        = 'categorical_crossentropy'    #Loss giving gradients
metrics     = ['accuracy', \
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
if targetData == 'cifar10':
    (trainingSetTarget, trainingLabelsTarget),(testSetTarget, testLabelsTarget) = \
            tf.keras.datasets.cifar10.load_data()
elif targetData == 'cifar100':
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
trainingLabelsSource = tf.keras.utils.to_categorical(trainingLabelsSource, 10)
testSetSource    = testSetSource/255.
testLabelsSource = tf.keras.utils.to_categorical(testLabelsSource, 10)

trainingSetTarget = trainingSetTarget/255.
if targetData == 'cifar100':
    trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, 100)
else:
    trainingLabelsTarget = tf.keras.utils.to_categorical(trainingLabelsTarget, 10)
testSetTarget = testSetTarget/255.
if targetData == 'cifar100':
    testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, 100)
else:
    testLabelsTarget = tf.keras.utils.to_categorical(testLabelsTarget, 10)

inputPlaceholderSource = tf.keras.Input([32, 32, 3], name='inputHolderSource')
inputPlaceholderTarget = tf.keras.Input([32, 32, 3], name='inputHolderTarget')
print("Normalizing data and creating one-hots : done\n")


#################################################
####### Building source and target models #######
#################################################

from quanta_models import createModel

print("Building source and target models : start")

def buildModels():
    NNSource     = tf.keras.Model(
        inputs=inputPlaceholderSource,
        outputs=createModel(
            placeholder=inputPlaceholderSource,
            outputSize=10,
            trsf_layer=trsf_layer)
        ) 
    NNSourceCopy = tf.keras.Model(
        inputs=inputPlaceholderTarget,
        outputs=createModel(
            placeholder=inputPlaceholderTarget,
            outputSize=10,
            trsf_layer=trsf_layer)
        )

    if targetData == 'cifar100':
        NNTarget  = tf.keras.Model(
            inputs=inputPlaceholderTarget, 
            outputs=createModel(
                placeholder=inputPlaceholderTarget,
                outputSize=100,
                trsf_layer=trsf_layer,
                sourceModel=NNSourceCopy)
            ) 
        NNWitness = tf.keras.Model(
            inputs=inputPlaceholderTarget,
            outputs=createModel(
                placeholder=inputPlaceholderTarget, 
                outputSize=100, 
                trsf_layer=trsf_layer)
            )
    else:
        NNTarget  = tf.keras.Model(
            inputs=inputPlaceholderTarget,
            outputs=createModel(
                placeholder=inputPlaceholderTarget,
                outputSize=10,
                trsf_layer=trsf_layer,
                sourceModel=NNSourceCopy)
            ) 
        NNWitness = tf.keras.Model(
            inputs=inputPlaceholderTarget,
            outputs=createModel(
                placeholder=inputPlaceholderTarget,
                outputSize=10,
                trsf_layer=trsf_layer)
            )

    for l in NNSourceCopy.layers: l.trainable=False
    NNSource.compile(optimizer, loss, metrics)
    NNTarget.compile(optimizer, loss, metrics)
    NNWitness.compile(optimizer, loss, metrics)

    #
    NotLoadSource  = True
    NotLoadTarget  = True
    NotLoadWitness = True
    return ((NNSource, NNSourceCopy, NNTarget, NNWitness),\
              (NotLoadSource, NotLoadTarget, NotLoadWitness))

(NNSource, NNSourceCopy, NNTarget, NNWitness),(NotLoadSource, NotLoadTarget, NotLoadWitness) = \
                  buildModels()
print("Building source and target models : done\n")


#################################################
######### Model training and evaluation #########
#################################################

from typing import cast
from QuantaLayer import QuantaLayer

# Callback for exporting accuracy during training
targetMetrics = []
recordTargetMetrics = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch,logs: targetMetrics.append(
        NNTarget.evaluate(testSetTarget, testLabelsTarget)))

# Data augmentation generator
dataAugmentationGenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,         #set input mean to 0 over the dataset
    samplewise_center=False,          #set each sample mean to 0
    featurewise_std_normalization=False,     #divide inputs by std of the dataset
    samplewise_std_normalization=False,      #divide each input by its std
    zca_whitening=False,            #apply ZCA whitening
    zca_epsilon=1e-06,              #epsilon for ZCA whitening
    rotation_range=45,              #randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,          #randomly shifting image horizontally
    height_shift_range=0.1,         #randomly shifting image vertically
    shear_range=0.1,                #set range for random shear
    zoom_range=0.1,                 #set range for random zoom
    channel_shift_range=0.,         #set range for random channel shifts
    fill_mode='nearest',            #fillmode for image manipulation
    cval=0.,                        #value used for fill_mode = "constant"
    horizontal_flip=True,           #randomly flip images horizontally
    vertical_flip=True,             #randomly flip images vertially
    rescale=None,
    preprocessing_function=None,
    data_format=None
)

def train(modelName, dataAugmentation=False, fromPreviousTraining=False): 
    # modelName : T for target, S for source, W for witness
    if (modelName == 'T'):
        print("Training target model...")
        NNSourceCopy.load_weights('./NNSource_w.h5')
        model = NNTarget
        trainingSet    = trainingSetTarget
        trainingLabels = trainingLabelsTarget
        weights = "./NNTarget_w.h5"
        noe = numberOfEpochsTarget
        cb  = [recordTargetMetrics]
        for i in range(len(model.layers)):
            if (type(model.layers[i]).__name__ == "QuantaLayer"):
                cb.append(cast(QuantaLayer, model.layers[i]).getCustomCallback())
        bc  = batchSizeTarget
        NotLoadTarget=False

    elif (modelName == 'S'):
        print("Training source model...")
        model = NNSource
        trainingSet    = trainingSetSource
        trainingLabels = trainingLabelsSource
        weights = "./NNSource_w.h5"
        cb  = []
        noe = numberOfEpochsSource
        bc  = batchSizeSource
        NotLoadSource=False
    else :
        print("Training witness model...")
        model = NNWitness
        trainingSet    = trainingSetTarget
        trainingLabels = trainingLabelsTarget
        weights = "./NNWitness_w.h5"
        cb  = []
        noe = numberOfEpochsWitness
        bc  = batchSizeSource
        NotLoadWitness=False

    if dataAugmentation: 
        dataAugmentationGenerator.fit(trainingSet, augment=True)
        if fromPreviousTraining: model.load_weights(weights)
        model.fit(
                dataAugmentationGenerator.flow(
                        trainingSet, trainingLabels, batch_size=bc), epochs=noe, callbacks=cb)
    else: 
        if fromPreviousTraining: model.load_weights(weights)
        model.fit(trainingSet, trainingLabels, epochs=noe, callbacks=cb, batch_size=bc)

    print("Saving model parameters")
    model.save_weights(weights)
    return

def test(modelName): 
    if (modelName == 'T'):
        print("Testing target model...")
        NNSourceCopy.load_weights("./NNSource_w.h5")
        model = NNTarget
        testSet    = testSetTarget
        testLabels = testLabelsTarget
        weights = "./NNTarget_w.h5"
        snl     = NotLoadTarget

    elif (modelName == 'S'):
        print("Testing source model...")
        model = NNSource
        testSet    = testSetSource
        testLabels = testLabelsSource
        weights = "./NNSource_w.h5"
        snl     = NotLoadWitness
    else :
        print("Testing witness model...")
        model = NNWitness
        testSet    = testSetTarget
        testLabels = testLabelsTarget
        weights = "./NNWitness_w.h5"
        snl     = NotLoadSource

    if snl: model.load_weights(weights)

    metrics = model.evaluate(testSet, testLabels)

    f = open(outputDir+'/metrics'+str(currentRun)+str(modelName)+'.txt', 'w')
    f.write(str(metrics)+'\n')
    f.close()
    print("Final testing accuracy :" +str(metrics[1])+"\n")
    return metrics[1] # accuracy


#################################################
############### Exporting metrics ###############
#################################################

def export_expe_summary(NNTarget, target_task, src_accuracy, target_accuracy):
    f = open(outputDir+'/expe_summary.txt', 'a')
    export  = 'Target task: '  + targetData + '\n'
    export += '(Eta: '+str(eta)+ ')\n'
    export += 'Target task' + str(target_task) + '\nSource model accuracy :' + \
                     str(float(src_accuracy)) + \
                     '\nTarget model accuracy: ' + str(target_accuracy) + \
                     '\nTarget model summary:\n'
    f.write(export)
    NNTarget.summary(line_length=80, print_fn=lambda x: f.write(x + '\n'))
    f.close()
    return


#################################################
##################### Main ######################
#################################################

def main1():
    print("Target task: ", targetData)

    print('Creating output directory')
    if not os.path.exists(outputDir): os.makedirs(outputDir)    

    print('Starting run ', currentRun)

    train('T', augmentData, fromPreviousTraining)

    #TODO
    #writeFactors(quantas, currentRun, raw=False)
    #writeFactors(raw_quantas, currentRun, raw=True)

    if currentRun == 0:
        export_expe_summary(NNTarget, targetData, test('S'), test('T'))

    f = open(outputDir+'/all_target_metrics_' + str(currentRun) + '.txt','a')
    for i in range(0, len(targetMetrics)): f.write(str(targetMetrics[i])+'\n')
    f.close()
    return

if __name__=="__main__":
    main1()
