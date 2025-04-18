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
from utils.datasets import load_dataset
from models import build_and_compile_model, load_model
from utils.test_and_train import get_training_config, train, test, reset_metrics, \
    export_quanta_metrics, export_quanta_from_model
from utils.transfer_argparser import parse_arguments


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

def transfer(arguments, current_run=1, all_at_once=False):
    """
    Main function to experiment gradual transfer with Quanta.
    
    Args:
        arguments:
            Argparser that contains all needed arguments.
        current_run:
            The number of the run.
            Default to 1.
        all_at_once:
            A boolean to indicate if all conv2d and dense layers have to be transfered.
            Default at True.
    """

    # Loading input data
    source_dataset = load_dataset(arguments.source_task, arguments.nb_of_samples)
    target_dataset = load_dataset(arguments.target_task, arguments.nb_of_samples)

    # Get tensorflow training configuration
    training_config = get_training_config(learning_rate=2e-4, weight_decay=1e-6)

    # Building and compiling models
    source_model = build_and_compile_model(
        model_name = "source",
        input_shape = source_dataset['input_shape'],
        output_shape = source_dataset['output_shape'],
        optimizer = training_config['optimizer'],
        loss = training_config['loss'],
        metrics = training_config['metrics'],
        trainable = False,
        weights_model_path=arguments.source_weights,
        all_at_once=all_at_once)
    if arguments.train_from_previous_training is None:
        target_model = build_and_compile_model(
            model_name = "target",
            input_shape = target_dataset['input_shape'],
            output_shape = target_dataset['output_shape'],
            optimizer = training_config['optimizer'],
            loss = training_config['loss'],
            metrics = training_config['metrics'],
            augment_data = arguments.augment_data,
            layer_to_transfer = arguments.layer_to_transfer,
            source_model = source_model,
            all_at_once=all_at_once)
    else:
        target_model = load_model(arguments.train_from_previous_training, model_name="target")

    if arguments.show_only_build:
        source_model.summary()
        target_model.summary()
        sys.exit(0)

    # Model training and evaluation
    testing_metrics_of_source = test(
        model = source_model,
        test_set = source_dataset['test_set'],
        test_labels = source_dataset['test_labels'])
    # Important to reset states of each used metric as they are shared by both models
    # Potentially, be also carefull about that point :
    # https://stackoverflow.com/questions/65923011/keras-tensoflow-full-reset
    reset_metrics(training_config['metrics'])
    training_metrics_of_target = train(
        model = target_model,
        training_set = target_dataset['training_set'],
        training_labels = target_dataset['training_labels'],
        nb_of_epoch = arguments.nb_of_epochs)
    testing_metrics_of_target = test(
        model = target_model,
        test_set = target_dataset['test_set'],
        test_labels = target_dataset['test_labels'])

    #Exporting metrics
    print("Export all metrics and data : start")
    export_quanta_metrics(
        arguments.output_dir,
        arguments.source_task,
        arguments.target_task,
        current_run,
        arguments.layer_to_transfer,
        target_model,
        training_metrics_of_target,
        False,
        "/training_metrics_of_")
    export_quanta_metrics(
        arguments.output_dir,
        arguments.source_task,
        arguments.target_task,
        current_run,
        arguments.layer_to_transfer,
        target_model,
        testing_metrics_of_target,
        True,
        "/testing_metrics_of_")
    export_quanta_metrics(
        arguments.output_dir,
        arguments.source_task,
        arguments.target_task,
        current_run,
        arguments.layer_to_transfer,
        source_model,
        testing_metrics_of_source,
        True,
        "/testing_metrics_of_")
    export_quanta_from_model(
        arguments.output_dir,
        arguments.source_task,
        arguments.target_task,
        current_run,
        arguments.layer_to_transfer,
        target_model
    )
    print("Export all metrics and data : done\n")

    print("Final testing categorical accuracy of source :" +
        str(testing_metrics_of_source['categorical_accuracy']))
    print("Final testing categorical accuracy of target :" +
        str(testing_metrics_of_target['categorical_accuracy']))

if __name__ == "__main__":
    # Parse all arguments of quanta evaluation tool
    args = parse_arguments()
    # Set global determinism if seed value is well defined
    if args.seed is not None and args.seed >= 0:
        set_global_determinism(seed=args.seed)
    # Look at layer_to_transfer value to determine the way to do the transfer
    if args.layer_to_transfer < 0 :
        for run in range(args.nb_of_runs):
            print("\t All layers at once - Run n°" + str(run+1) + "\n")
            transfer(args, run, all_at_once=True)
    else:
        for run in range(args.nb_of_runs):
            print("\t Layer " + str(args.layer_to_transfer) + " - Run n°" + str(run+1) + "\n")
            transfer(args, run, all_at_once=False)
