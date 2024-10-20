"""Module to parse arguments of quanta evaluation tool."""

import os
import argparse


#################################################
############# Custom action classes #############
#################################################

class CheckPositive(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values < 0:
            parser.error("{0} has to be positive, but {1} has been given.".format(option_string, values))
        setattr(namespace, self.dest, values)

class CheckStrictlyPositive(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values <= 0:
            parser.error("{0} has to be strictly positive, but {1} has been given.".format(option_string, values))
        setattr(namespace, self.dest, values)

class CheckKerasModelExist(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            parser.error("{0}: {1} file not found.".format(option_string, values))
        if not values.endswith('.keras'):
            parser.error("{0}: {1} is not a keras model.".format(option_string, values))
        setattr(namespace, self.dest, values)

class CheckWeightsModelExist(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            parser.error("{0}: {1} file not found.".format(option_string, values))
        if not values.endswith('.h5'):
            parser.error("{0}: {1} is not a weight model.".format(option_string, values))
        setattr(namespace, self.dest, values)


#################################################
############### Parsing arguments ###############
#################################################

def parse_arguments():
    """Function to parse arguments to experiment with Quanta"""

    parser = argparse.ArgumentParser(description="Quanta evaluation tool")
    parser.add_argument("-o", "--output_dir", default="./expe")
    parser.add_argument("-s", "--source_task", choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'], required=True)
    parser.add_argument("-sw", "--source_weights", action=CheckWeightsModelExist, default=None, type=str, required=True)
    parser.add_argument("-t", "--target_task", choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'], required=True)
    parser.add_argument("-l", "--layer_to_transfer", type=int, required=True)
    parser.add_argument("-r", "--nb_of_runs", action=CheckStrictlyPositive, default=1, type=int)
    parser.add_argument("--nb_of_target_epochs", action=CheckStrictlyPositive, default=1, type=int)
    parser.add_argument("--nb_of_target_samples", action=CheckStrictlyPositive, default=0, type=int)
    parser.add_argument("--seed", action=CheckPositive, default=-1, type=int)
    parser.add_argument("--augment_data", action='store_true')
    parser.add_argument("--train_from_previous_training", action=CheckKerasModelExist, default=None, type=str)
    parser.add_argument("--show_only_build", action='store_true')
    args=parser.parse_args()

    print("\nPython parameters :")
    print("\t Output directory : ", args.output_dir)
    if not os.path.exists(args.output_dir):
        print('\t\t Creating non-existing output directory')
        os.makedirs(args.output_dir)
    print("\t Source task : ", args.source_task)
    print("\t Source weights : ", args.source_weights)
    print("\t Target task : ", args.target_task)
    print("\t Layer to transfer : ", args.layer_to_transfer)
    print("\t Nb of runs : ", args.nb_of_runs)
    print("\t Seed : ", args.seed)
    print("\t Nb of target epochs : ", args.nb_of_target_epochs)
    print("\t Nb of samples : ", args.nb_of_target_samples)
    print("\t Augment data : ", args.augment_data)
    print("\t Train from previous training : ", args.train_from_previous_training)
    print("\t Show only build : ", args.show_only_build)
    print("\n")
    return args