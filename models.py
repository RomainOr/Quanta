"""Collection of functions to build models used with tensorflow."""

import tensorflow as tf

from quanta_layer import QuantaLayer


#################################################
#### Building blocks of convolutionnal layers ###
#################################################

def build_block(
        inputs,
        size,
        layer_to_tranfer=None,
        outputs_of_block_source_layer=None,
        all_at_once=False):
    """
    Build a block of two convolutionnal layers within a tensorflow model of given size. \n
    Arguments *layer_to_tranfer* and *outputs_of_block_source_layer* can respectively indicate 
    which layer has to be transfered through a Quanta layer.

    Args:
        inputs:
            The inputs built with tensorflow.
        size:
            The size of the two conv2d layer used in this model.
        layer_to_tranfer: 
            The number of the layer to transfer that could be a dense layer or a conv2d layer.
            Default at None.
        outputs_of_block_source_layer: 
            A list to build a quanta layer with respect to the layer to transfer.
            Default at None.
        all_at_once: 
            A boolean to indicate if all conv2d and dense layers have to be transfered.
            Default at True.

    Returns:
        The block built from inputs with respect to the other arguments.
    """

    x = inputs
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if all_at_once or \
            (layer_to_tranfer is not None and outputs_of_block_source_layer is not None
             and layer_to_tranfer % 2 == 0):
        x = QuantaLayer('elu')([outputs_of_block_source_layer[0], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if all_at_once or \
            (layer_to_tranfer is not None and outputs_of_block_source_layer is not None
             and layer_to_tranfer % 2 == 1):
        x = QuantaLayer('elu')([outputs_of_block_source_layer[1], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return x


#################################################
####### Create full model using convblocks ######
#################################################

def create_model(
        inputs,
        output_shape,
        layer_to_transfer=None,
        source_model=None,
        augment_data=False,
        all_at_once=False):
    """
    Build a model made from convolutionnal blocks from *build_block* function.

    Args:
        inputs:
            The inputs built with tensorflow.
        output_shape:
            The output shape of the last layer used in this model.
        layer_to_tranfer: 
            The number of the layer to transfer that could be a dense layer or a conv2d layer.
            Default at None.
        source_model: 
            A tensorflow model to get the layer to transfer?
            Default at None.
        augment_data: 
            A boolean to add layers to augment data or not in the model.
            Default at None.
        all_at_once: 
            A boolean to indicate if all conv2d and dense layers have to be transfered.
            Default at True.

    Returns:
        The whole built model.
    """

    outputs_of_block_source_layer = []
    if source_model is not None:
        outputs_of_block_source_layer = [
            layer.output
            for layer in source_model.layers
            if type(layer).__name__ in ('Conv2D', 'Dense')]

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    if augment_data:
        x = tf.keras.layers.RandomFlip(
            mode="horizontal_and_vertical")(x)
        x = tf.keras.layers.RandomTranslation(
            height_factor=0.1,
            width_factor=0.1)(x)
        x = tf.keras.layers.RandomRotation(
            factor=0.1)(x)
        x = tf.keras.layers.RandomZoom(
            height_factor=0.1,
            width_factor=None)(x)

    if (all_at_once and source_model is not None) or \
            (outputs_of_block_source_layer != [] and (layer_to_transfer in (0, 1))):
        x = build_block(
            x,
            64,
            layer_to_transfer,
            [outputs_of_block_source_layer[0], outputs_of_block_source_layer[1]],
            all_at_once)
    else:
        x = build_block(x, 64)

    if (all_at_once and source_model is not None) or \
            (outputs_of_block_source_layer != [] and (layer_to_transfer in (2, 3))):
        x = build_block(
            x,
            128,
            layer_to_transfer,
            [outputs_of_block_source_layer[2], outputs_of_block_source_layer[3]],
            all_at_once)
    else:
        x = build_block(x, 128)

    if (all_at_once and source_model is not None) or \
            (outputs_of_block_source_layer != [] and (layer_to_transfer in (4, 5))):
        x = build_block(
            x,
            512,
            layer_to_transfer,
            [outputs_of_block_source_layer[4], outputs_of_block_source_layer[5]],
            all_at_once)
    else:
        x = build_block(x, 512)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (all_at_once and source_model is not None) or \
            (outputs_of_block_source_layer != [] and layer_to_transfer == 6):
        x = QuantaLayer('elu')([outputs_of_block_source_layer[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

    return x


#################################################
######### Configure models for training #########
#################################################

def build_and_compile_model(
        model_name,
        input_shape,
        output_shape,
        optimizer,
        loss,
        metrics,
        augment_data=False,
        layer_to_transfer=None,
        source_model=None,
        trainable=True,
        weights_model_path=None,
        all_at_once=False):
    """
    Build and compile one model.

    Args:
        model_name:
            A personalized name givent to the built model.
        input_shape:
            The input shape of the model.
        output_shape:
            The output shape of the last layer used in this model.
        optimizer:
            When compiling the model, string or optimizer instance to train the model.
        loss:
            When compiling the model, string or loss function instance to train the model.
        metrics:
            List of metrics to be evaluated by the model during training and testing.
        augment_data: 
            A boolean to add layers to augment data or not in the model.
            Default at None.
        layer_to_tranfer: 
            The number of the layer to transfer that could be a dense layer or a conv2d layer.
            Default at None.
        source_model: 
            A tensorflow model to get the layer to transfer?
            Default at None.
        trainable:
            A boolean to indicate whether the model is trainable or not.
            Default at True.
        weights_model_path:
            A path to a '.h5' file to load weights withion the built model.
            Default at None.
        all_at_once: 
            A boolean to indicate if all conv2d and dense layers have to be transfered.
            Default at True.

    Returns:
        The whole model built and compiled, ready to be trained or evaluated.
    """

    print("Building " + model_name + " model : start")
    if source_model is not None and layer_to_transfer is not None:
        inputs = source_model.get_layer(index=0).output
    else:
        inputs = tf.keras.Input(input_shape)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=create_model(
            inputs=inputs,
            output_shape=output_shape,
            layer_to_transfer=layer_to_transfer,
            source_model=source_model,
            augment_data=augment_data,
            all_at_once=all_at_once)
    )
    if not trainable:
        for l in model.layers:
            l.trainable = False
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Building " + model_name + " model : done\n")

    if weights_model_path is not None:
        print("Loading " + model_name + " weights : start")
        model.load_weights(weights_model_path)
        print("Loading " + model_name + " weights : done\n")

    model._name = model_name
    return model


def load_target_model(model_path, model_name="target"):
    """Load a saved target model."""

    print("Loading " + model_name + " model : start")
    model = tf.keras.models.load_model(model_path)
    print("Loading " + model_name + " model : done\n")
    model._name = model_name
    return model
