"""Module to get custom action classes for argparser."""

import os
import argparse

#################################################
############# Custom action classes #############
#################################################

class CheckPositive(argparse.Action):
    """Action class of Argparse to check if the given int is positive."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values < 0:
            parser.error(
                f"{option_string} has to be positive, but {values} has been given.")
        setattr(namespace, self.dest, values)


class CheckStrictlyPositive(argparse.Action):
    """Action class of Argparse to check if the given int is strictly positive."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values <= 0:
            parser.error(
                f"{option_string} has to be strictly positive, but {values} has been given.")
        setattr(namespace, self.dest, values)


class CheckKerasModelExist(argparse.Action):
    """Action class of Argparse to check if the given value is an existing keral model."""

    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            parser.error(f"{option_string}: {values} file not found.")
        if not values.endswith('.keras'):
            parser.error(f"{option_string}: {values} is not a keras model.")
        setattr(namespace, self.dest, values)


class CheckWeightsModelExist(argparse.Action):
    """Action class of Argparse to check if the given value is an existing weight model."""

    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            parser.error(f"{option_string}: {values} file not found.")
        if not values.endswith('.h5'):
            parser.error(f"{option_string}: {values} is not a weight model.")
        setattr(namespace, self.dest, values)
