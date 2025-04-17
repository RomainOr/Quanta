"""Module to parse arguments of quanta evaluation tool."""

import os
import argparse

from utils.custom_action_classes import CheckPositive, \
    CheckStrictlyPositive, CheckKerasModelExist

#################################################
############### Parsing arguments ###############
#################################################

def parse_arguments():
    """
    Function to parse arguments to experiment with Quanta.

    Returns:
        A python object built with ArgumentParser that contains all arguments.
    """

    parser = argparse.ArgumentParser(description="Quanta evaluation tool")
    parser.add_argument("-o", "--output_dir",
                        default="./expe")
    parser.add_argument("-s", "--source_task",
                        choices=['cifar10', 'cifar100', 'mnist', 'fashion_mnist'], required=True)
    parser.add_argument("--nb_of_epochs",
                        action=CheckStrictlyPositive, default=1, type=int)
    parser.add_argument("--nb_of_samples",
                        action=CheckStrictlyPositive, default=0, type=int)
    parser.add_argument("--seed",
                        action=CheckPositive, default=-1, type=int)
    parser.add_argument("--augment_data",
                        action='store_true')
    parser.add_argument("--train_from_previous_training",
                        action=CheckKerasModelExist, default=None, type=str)
    parser.add_argument("--show_only_build", action='store_true')
    args = parser.parse_args()

    print("\nPython parameters :")
    print("\t Output directory : ", args.output_dir)
    if not os.path.exists(args.output_dir):
        print('\t\t Creating non-existing output directory')
        os.makedirs(args.output_dir)
    print("\t Source task : ", args.source_task)
    print("\t Nb of epochs : ", args.nb_of_epochs)
    print("\t Nb of samples : ", args.nb_of_samples)
    print("\t Seed : ", args.seed)
    print("\t Augment data : ", args.augment_data)
    print("\t Train from previous training : ",
          args.train_from_previous_training)
    print("\t Show only build : ", args.show_only_build)
    print("\n")
    return args
