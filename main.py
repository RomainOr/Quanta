"""Main module"""

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Less verbosity

import sys
import random
import numpy as np
import tensorflow as tf
from quanta_layer import QuantaLayer
from models import build_and_compile_model
from typing import cast


#################################################
####### Managing arguments and parameters #######
#################################################
# TODO : Manage exceptions and defaults wrt shell script

output_dir = sys.argv[1]
current_run = int(sys.argv[2])
layer_to_transfer = int(sys.argv[3])
target_task = sys.argv[4]
input_seed = int(sys.argv[5])

print("Python parameters :")
print("\t Output directory : ", output_dir)
if not os.path.exists(output_dir):
    print('\t\t Creating non-existing output directory')
    os.makedirs(output_dir)
print("\t Current run : ", current_run)
print("\t Layer : ", layer_to_transfer)
print("\t Target task : ", target_task)
print("\t Seed : ", input_seed)
print("\n")


#################################################
##### Set up Determinism or not with QUANTA #####
#################################################

def set_global_determinism(seed):
    """Wrapper to globally control determinism through a seed value."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# Call the above function with seed value if well defined
if input_seed is not None and input_seed > 0:
    set_global_determinism(seed=input_seed)

# User defined parameters
NB_OF_SAMPLES = 320

NB_OF_SOURCE_EPOCHS = 1
NB_OF_TARGET_EPOCHS = 1

OPTIMIZER = tf.keras.optimizers.Adam(
    learning_rate=2e-4,
    weight_decay=1e-6)
LOSS = tf.keras.losses.CategoricalCrossentropy()
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.FalseNegatives(),
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.TopKCategoricalAccuracy(k=3)]

AUGMENT_DATA = False
TRAIN_FROM_PREVIOUS_TRAINING = False


#################################################
############## Loading input data ###############
#################################################

print("Loading Data : start")
source_output_shape = 10
(training_set_source, training_labels_source), (test_set_source, test_labels_source) = \
    tf.keras.datasets.cifar10.load_data()

target_output_shape = -1
if target_task == 'cifar100':
    (training_set_target, training_labels_target), (test_set_target, test_labels_target) = \
        tf.keras.datasets.cifar100.load_data()
    target_output_shape = 100
else:
    (training_set_target, training_labels_target), (test_set_target, test_labels_target) = \
        tf.keras.datasets.cifar10.load_data()
    target_output_shape = 10
input_shape = training_set_target.shape[1:]

training_set_source = training_set_source[:NB_OF_SAMPLES]
training_labels_source = training_labels_source[:NB_OF_SAMPLES]
test_set_source = test_set_source[:NB_OF_SAMPLES]
test_labels_source = test_labels_source[:NB_OF_SAMPLES]

training_set_target = training_set_target[:NB_OF_SAMPLES]
training_labels_target = training_labels_target[:NB_OF_SAMPLES]
test_set_target = test_set_target[:NB_OF_SAMPLES]
test_labels_target = test_labels_target[:NB_OF_SAMPLES]
print("Loading Data : done\n")


#################################################
############### Creating one-hots ###############
#################################################

print("Creating one-hots : start")
training_labels_source = tf.keras.utils.to_categorical(training_labels_source)
test_labels_source = tf.keras.utils.to_categorical(test_labels_source)

training_labels_target = tf.keras.utils.to_categorical(training_labels_target)
test_labels_target = tf.keras.utils.to_categorical(test_labels_target)
print("Creating one-hots : done\n")


#################################################
######### Building and compiling models #########
#################################################

source_model = build_and_compile_model(
    model_name = "source",
    input_shape = input_shape,
    output_shape = source_output_shape,
    optimizer = OPTIMIZER,
    loss = LOSS,
    metrics = METRICS,
    trainable = False,
    weights_path="./SourceModel.weights.h5")
target_model = build_and_compile_model(
    model_name = "target",
    input_shape = input_shape,
    output_shape = target_output_shape,
    optimizer = OPTIMIZER,
    loss = LOSS,
    metrics = METRICS,
    augment_data = AUGMENT_DATA,
    layer_to_transfer = layer_to_transfer,
    source_model = source_model)


#################################################
######### Model training and evaluation #########
#################################################

def train(model, training_set, training_labels, nb_of_epoch, 
          save_weights_path, train_from_previous_training=False):
    """Train a model."""

    print("Training " + model._name + " model : start")
    training_metrics = {}
    cb = [tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs:
        training_metrics.update({epoch: logs})
    )]
    for idx, layer in enumerate(model.layers):
        if type(layer).__name__ == "QuantaLayer":
            cb.append(cast(QuantaLayer, layer).get_custom_callback(idx))

    if train_from_previous_training:
        model.load_weights(save_weights_path)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (training_set, training_labels))
    train_dataset = train_dataset.batch(32).map(lambda x, y: (x, y))

    model.fit(train_dataset, epochs=nb_of_epoch, callbacks=cb)
    print("Training " + model._name + " model : done\n")

    print("Saving model parameters : start")
    model.save_weights(save_weights_path)
    print("Saving model parameters : done\n")
    return training_metrics


def test(model, test_set, test_labels, weights):
    """Test a model"""
    
    print("Testing " + model._name + " model : start")
    model.load_weights(weights)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_set, test_labels))
    test_dataset = test_dataset.batch(32).map(lambda x, y: (x, y))
    testing_metrics = model.evaluate(test_dataset, return_dict=True)
    print("Testing " + model._name + " model : done\n")
    return testing_metrics


testing_metrics_of_source = test(
    model = source_model,
    test_set = test_set_source,
    test_labels = test_labels_source,
    weights = "./SourceModel.weights.h5")

# Important to reset states of each used metric as they are shared by both models
for metric in METRICS:
    cast(tf.keras.metrics.Metric, metric).reset_state()
# Potentially, be also carefull about that point :
# https://stackoverflow.com/questions/65923011/keras-tensoflow-full-reset

training_metrics_of_target = train(
    model = target_model,
    training_set = training_set_target,
    training_labels = training_labels_target,
    nb_of_epoch = NB_OF_TARGET_EPOCHS,
    save_weights_path = output_dir + "/TargetModel.weights.h5",
    train_from_previous_training = TRAIN_FROM_PREVIOUS_TRAINING)
testing_metrics_of_target = test(
    model = target_model,
    test_set = test_set_target,
    test_labels = test_labels_target,
    weights = output_dir + "/TargetModel.weights.h5")


#################################################
############### Exporting metrics ###############
#################################################

def export_metrics(
    output_dir,
    current_run,
    layer_to_transfer,
    model,
    metrics_of_model,
    save_model,
    string
    ):
    '''Export metrics'''
    tmp_string = model._name + '_model_run_'+str(current_run)+'_layer_'+str(layer_to_transfer)
    f = open(
        file=output_dir+'/' + string + tmp_string + '.jsonl',
        mode='a',
        encoding='UTF-8'
    )
    f.write(str(metrics_of_model)+'\n')
    f.close()

    if save_model:
        model.save(output_dir + '/' + tmp_string + '.keras')

export_metrics(output_dir, current_run, layer_to_transfer, 
               target_model, training_metrics_of_target, False, "training_metrics_of_")
export_metrics(output_dir, current_run, layer_to_transfer, 
               target_model, testing_metrics_of_target, True, "testing_metrics_of_")
export_metrics(output_dir, current_run, layer_to_transfer, 
               source_model, testing_metrics_of_source, True, "testing_metrics_of_")

print("Final testing categorical accuracy of source :" +
    str(testing_metrics_of_source['categorical_accuracy']))
print("Final testing categorical accuracy of target :" +
    str(testing_metrics_of_target['categorical_accuracy']))
