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

import random
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from models import build_and_compile_model, load_target_model
from test_and_train import get_training_config, train, test, reset_metrics, \
    export_metrics, export_quanta_from_model
from quanta_arguments_parser import parse_arguments


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

def gradual_transfer(arguments, current_run=1):
    """Main function to experiment gradual transfer with Quanta"""

    # Loading input data
    source_dataset = load_dataset(arguments.source_task, arguments.nb_of_target_samples)
    target_dataset = load_dataset(arguments.target_task, arguments.nb_of_target_samples)

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
        weights_model_path=arguments.source_weights)
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
            source_model = source_model)
    else:
        target_model = load_target_model(arguments.train_from_previous_training)

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
        nb_of_epoch = arguments.nb_of_target_epochs)
    testing_metrics_of_target = test(
        model = target_model,
        test_set = target_dataset['test_set'],
        test_labels = target_dataset['test_labels'])

    #Exporting metrics
    print("Export all metrics and data : start")
    export_metrics(
        arguments.output_dir,
        current_run,
        arguments.layer_to_transfer,
        target_model,
        training_metrics_of_target,
        False, 
        "/training_metrics_of_")
    export_metrics(
        arguments.output_dir,
        current_run,
        arguments.layer_to_transfer,
        target_model,
        testing_metrics_of_target,
        True,
        "/testing_metrics_of_")
    export_metrics(
        arguments.output_dir,
        current_run,
        arguments.layer_to_transfer,
        source_model,
        testing_metrics_of_source,
        True,
        "/testing_metrics_of_")
    export_quanta_from_model(
        arguments.output_dir,
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
        print("Transfer all at once in development.")
    else:
        for run in range(args.nb_of_runs):
            print("\t Layer " + str(args.layer_to_transfer) + " - Run n°" + str(run+1) + "\n")
            gradual_transfer(args, run)
