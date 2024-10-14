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
from datasets import load_dataset
from models import build_and_compile_model
from test_and_train import get_training_config, train, test, reset_metrics, export_metrics


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


#################################################
############ Main function of module ############
#################################################


def gradual_transfer():
    """Main function to experiment gradual transfer with Quanta"""

    # Managing arguments and parameters
    # TODO : Manage exceptions and defaults wrt shell script with argparse
    output_dir = sys.argv[1]
    current_run = int(sys.argv[2])
    layer_to_transfer = int(sys.argv[3])
    target_task = sys.argv[4]
    source_task = 'cifar10'
    input_seed = int(sys.argv[5])
    nb_of_samples = 320
    nb_of_target_epochs = 1
    augment_data = False
    train_from_previous_training = False

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

    # Call the above function with seed value if well defined
    if input_seed is not None and input_seed > 0:
        set_global_determinism(seed=input_seed)

    # Loading input data
    source_dataset = load_dataset(source_task, nb_of_samples)
    target_dataset = load_dataset(target_task, nb_of_samples)

    # Get tensorflow training configuration
    training_config = get_training_config()

    # Building and compiling models
    source_model = build_and_compile_model(
        model_name = "source",
        input_shape = source_dataset['input_shape'],
        output_shape = source_dataset['output_shape'],
        optimizer = training_config['optimizer'],
        loss = training_config['loss'],
        metrics = training_config['metrics'],
        trainable = False,
        weights_path="./SourceModel.weights.h5")
    target_model = build_and_compile_model(
        model_name = "target",
        input_shape = target_dataset['input_shape'],
        output_shape = target_dataset['output_shape'],
        optimizer = training_config['optimizer'],
        loss = training_config['loss'],
        metrics = training_config['metrics'],
        augment_data = augment_data,
        layer_to_transfer = layer_to_transfer,
        source_model = source_model)

    # Model training and evaluation
    testing_metrics_of_source = test(
        model = source_model,
        test_set = source_dataset['test_set'],
        test_labels = source_dataset['test_labels'],
        weights = "./SourceModel.weights.h5")
    # Important to reset states of each used metric as they are shared by both models
    # Potentially, be also carefull about that point :
    # https://stackoverflow.com/questions/65923011/keras-tensoflow-full-reset
    reset_metrics(training_config['metrics'])
    training_metrics_of_target = train(
        model = target_model,
        training_set = target_dataset['training_set'],
        training_labels = target_dataset['training_labels'],
        nb_of_epoch = nb_of_target_epochs,
        save_weights_path = output_dir + "/TargetModel.weights.h5",
        train_from_previous_training = train_from_previous_training)
    testing_metrics_of_target = test(
        model = target_model,
        test_set = target_dataset['test_set'],
        test_labels = target_dataset['test_labels'],
        weights = output_dir + "/TargetModel.weights.h5")

    #Exporting metrics
    export_metrics(output_dir, current_run, layer_to_transfer, 
                target_model, training_metrics_of_target, False, "/training_metrics_of_")
    export_metrics(output_dir, current_run, layer_to_transfer, 
                target_model, testing_metrics_of_target, True, "/testing_metrics_of_")
    export_metrics(output_dir, current_run, layer_to_transfer, 
                source_model, testing_metrics_of_source, True, "/testing_metrics_of_")

    print("Final testing categorical accuracy of source :" +
        str(testing_metrics_of_source['categorical_accuracy']))
    print("Final testing categorical accuracy of target :" +
        str(testing_metrics_of_target['categorical_accuracy']))

if __name__ == "__main__":
    gradual_transfer()