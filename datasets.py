"""Module to load the datasets"""

import tensorflow as tf


#################################################
############## Loading input data ###############
#################################################

def load_dataset(task, nb_of_samples=None, create_one_hots=True):
    """
    Load a dataset from tensorflow keras and return it with input and output shapes.

    Args:
        task: 
            A string expliciting the task.
        nb_of_samples: 
            Number of samples to have in the sets to reduce training time for example. 
            Default at None.
        create_one_hots: 
            A boolean to transform labels as one-hots.
            Default at True.

    Returns:
        Dictionnary with training and test sets and the related labels, 
        the input and output shapes.

    Example:
        ```
        load_dataset('cifar10')
        load_dataset('cifar100', 320)
        load_dataset('mnist', 6400, True)
        ```
    """

    output_shape = -1
    if task == 'cifar100':
        print("Loading Data of (" + task + ") : start")
        (training_set, training_labels), (test_set, test_labels) = \
            tf.keras.datasets.cifar100.load_data()
        output_shape = 100
    elif task == 'cifar10':
        print("Loading Data of (" + task + ") : start")
        (training_set, training_labels), (test_set, test_labels) = \
            tf.keras.datasets.cifar10.load_data()
        output_shape = 10
    elif task == 'mnist':
        print("Loading Data of (" + task + ") : start")
        (training_set, training_labels), (test_set, test_labels) = \
            tf.keras.datasets.mnist.load_data()
        output_shape = 10
    elif task == 'fashion_mnist':
        print("Loading Data of (" + task + ") : start")
        (training_set, training_labels), (test_set, test_labels) = \
            tf.keras.datasets.fashion_mnist.load_data()
        output_shape = 10
    else:
        print("Loading Data of (cifar10) : start")
        (training_set, training_labels), (test_set, test_labels) = \
            tf.keras.datasets.cifar10.load_data()
        output_shape = 10

    input_shape = training_set.shape[1:]

    if nb_of_samples is not None and nb_of_samples > 0:
        training_set = training_set[:nb_of_samples]
        training_labels = training_labels[:nb_of_samples]
        test_set = test_set[:nb_of_samples]
        test_labels = test_labels[:nb_of_samples]

    print("Loading Data : done\n")

    if create_one_hots:
        print("Creating one-hots : start")
        training_labels = tf.keras.utils.to_categorical(training_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)
        print("Creating one-hots : done\n")

    return_result = {
        'output_shape': output_shape,
        'input_shape': input_shape,
        'training_set': training_set,
        'training_labels': training_labels,
        'test_set': test_set,
        'test_labels': test_labels
    }
    return return_result
