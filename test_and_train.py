"""Test and train module"""

import os
import json
import tensorflow as tf

from typing import cast
from quanta_layer import QuantaLayer


#################################################
##### Getter of tensorflow training config ######
#################################################

def get_training_config(learning_rate=2e-4, weight_decay=1e-6):
    """Return the optimizer, loss and metrics used to train or test a model."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate = learning_rate,
        weight_decay = weight_decay)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    return {'optimizer':optimizer, 'loss':loss, 'metrics':metrics}


#################################################
######### Model training and evaluation #########
#################################################

def train(model, training_set, training_labels, nb_of_epoch):
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

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (training_set, training_labels))
    train_dataset = train_dataset.batch(32).map(lambda x, y: (x, y))

    model.fit(train_dataset, epochs=nb_of_epoch, callbacks=cb)
    print("Training " + model._name + " model : done\n")
    return training_metrics


def test(model, test_set, test_labels):
    """Test a model"""

    print("Testing " + model._name + " model : start")
    test_dataset = tf.data.Dataset.from_tensor_slices((test_set, test_labels))
    test_dataset = test_dataset.batch(32).map(lambda x, y: (x, y))
    testing_metrics = model.evaluate(test_dataset, return_dict=True)
    print("Testing " + model._name + " model : done\n")
    return testing_metrics


#################################################
########### Reseting states of metrics ##########
#################################################

def reset_metrics(metrics):
    """Reset states of each used metric in metrics"""

    for metric in metrics:
        cast(tf.keras.metrics.Metric, metric).reset_state()


#################################################
############### Exporting metrics ###############
#################################################

def export_metrics(
        output_dir,
        source_task,
        target_task,
        current_run,
        layer_to_transfer,
        model,
        metrics_of_model,
        save_model,
        filename_begin):
    """Export metrics to a jsonl file."""

    layer_to_transfer_str = 'all' if layer_to_transfer < 0 else str(layer_to_transfer)
    filename_end = model._name + '_r_' + str(current_run) + '_l_'+ layer_to_transfer_str
    directory = output_dir + '/' + source_task + '_to_' + target_task
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(
        file=directory + filename_begin + filename_end + '.jsonl',
        mode='a',
        encoding='UTF-8'
    ) as f :
        json.dump(metrics_of_model, f, indent=4)
        f.write('\n')
        f.close()

    if save_model:
        model.save(directory + '/' + filename_end + '.keras')

def export_quanta_from_model(
        output_dir,
        source_task,
        target_task,
        current_run,
        layer_to_transfer,
        model):
    """Export quanta values to a jsonl file."""

    layer_to_transfer_str = 'all' if layer_to_transfer < 0 else str(layer_to_transfer)
    filename = '/quantas' + '_r_' + str(current_run) + '_l_'+ layer_to_transfer_str
    directory = output_dir + '/' + source_task + '_to_' + target_task
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(
        file=directory + filename + '.jsonl',
        mode='a',
        encoding='UTF-8'
    ) as f:
        quantas = {}
        for idx, layer in enumerate(model.layers):
            if type(layer).__name__ == 'QuantaLayer':
                quantas[str(idx)] = {
                    "quanta_weights": cast(QuantaLayer, layer).get_quanta_weights(),
                    "quantas" : cast(QuantaLayer, layer).get_quantas()
                }
        json.dump(quantas, f , indent=4)
        f.write('\n')
        f.close()
