"""Main module to train source model"""

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
from utils.datasets import load_dataset
from models import build_and_compile_model, load_model
from utils.test_and_train import get_training_config, train, test, reset_metrics, \
    export_source_metrics
from utils.train_source_model_argparser import parse_arguments


#################################################
##### Set up Determinism or not with QUANTA #####
#################################################

def set_global_determinism(seed):
    """
    Wrapper to globally control determinism through a seed value.
    
    Args:
        Seed:
            A positive int to use as a seed for numpy, random and tf prng.
    """

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

def train_source_model(arguments):
    """
    Main function to train a source model thta is then used with Quanta.
    
    Args:
        arguments:
            Argparser that contains all needed arguments.
    """

    # Loading input data
    source_dataset = load_dataset(arguments.source_task, arguments.nb_of_samples)

    # Get tensorflow training configuration
    training_config = get_training_config(learning_rate=2e-4, weight_decay=1e-6)

    # Building and compiling models
    if arguments.train_from_previous_training is None:
        source_model = build_and_compile_model(
            model_name = "source",
            input_shape = source_dataset['input_shape'],
            output_shape = source_dataset['output_shape'],
            optimizer = training_config['optimizer'],
            loss = training_config['loss'],
            metrics = training_config['metrics'])
    else:
        source_model = load_model(arguments.train_from_previous_training, model_name="source")

    if arguments.show_only_build:
        source_model.summary()
        sys.exit(0)

    # Model training and evaluation
    training_metrics_of_source = train(
        model = source_model,
        training_set = source_dataset['training_set'],
        training_labels = source_dataset['training_labels'],
        nb_of_epoch = arguments.nb_of_epochs)
    reset_metrics(training_config['metrics'])
    # Model training and evaluation
    testing_metrics_of_source = test(
        model = source_model,
        test_set = source_dataset['test_set'],
        test_labels = source_dataset['test_labels'])
    # Important to reset states of each used metric as they are shared by both models
    # Potentially, be also carefull about that point :
    # https://stackoverflow.com/questions/65923011/keras-tensoflow-full-reset
    reset_metrics(training_config['metrics'])

    #Exporting metrics
    print("Export all metrics and data : start")
    export_source_metrics(
        arguments.output_dir,
        arguments.source_task,
        source_model,
        training_metrics_of_source,
        True,
        "/training_metrics_of_")
    export_source_metrics(
        arguments.output_dir,
        arguments.source_task,
        source_model,
        testing_metrics_of_source,
        True,
        "/testing_metrics_of_")
    print("Export all metrics and data : done\n")

    print("Final testing categorical accuracy of source :" +
        str(testing_metrics_of_source['categorical_accuracy']))

if __name__ == "__main__":
    # Parse all arguments of quanta evaluation tool
    args = parse_arguments()
    # Set global determinism if seed value is well defined
    if args.seed is not None and args.seed >= 0:
        set_global_determinism(seed=args.seed)
    # Train a source model
    print("\t Source model training :\n")
    train_source_model(args)
