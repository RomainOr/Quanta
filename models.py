"""Collection of functions to build models used with tensorflow."""

import tensorflow as tf

from quanta_layer import QuantaLayer


#################################################
#### Building blocks of convolutionnal layers ###
#################################################

def build_block(inputs, size, layer_to_tranfer=None, outputs_of_block_source_layer=None):
    """Build a block of two convolutionnal layers within a tensorflow model of given size. \n
    Arguments *layer_to_tranfer* and *outputs_of_block_source_layer* can respectively indicate 
    which layer has to be transfered through a Quanta layer.
    """

    x = inputs
    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (layer_to_tranfer is not None and outputs_of_block_source_layer is not None
            and layer_to_tranfer % 2 == 0):
        x = QuantaLayer('elu')([outputs_of_block_source_layer[0], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(size, 3, padding='same')(x)
    if (layer_to_tranfer is not None and outputs_of_block_source_layer is not None
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

def create_model(inputs, size_of_outputs, layer_to_transfer=None, source_model=None, augment_data=False):
    """Build a model made from convolutionnal blocks from *build_block* function."""

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

    if (outputs_of_block_source_layer != [] and (layer_to_transfer in (0, 1))):
        x = build_block(x, 64, layer_to_transfer, [
            outputs_of_block_source_layer[0], outputs_of_block_source_layer[1]])
    else:
        x = build_block(x, 64)

    if (outputs_of_block_source_layer != [] and (layer_to_transfer in (2, 3))):
        x = build_block(x, 128, layer_to_transfer, [
            outputs_of_block_source_layer[2], outputs_of_block_source_layer[3]])
    else:
        x = build_block(x, 128)

    if (outputs_of_block_source_layer != [] and (layer_to_transfer in (4, 5))):
        x = build_block(x, 512, layer_to_transfer, [
            outputs_of_block_source_layer[4], outputs_of_block_source_layer[5]])
    else:
        x = build_block(x, 512)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (outputs_of_block_source_layer != [] and layer_to_transfer == 6):
        x = QuantaLayer('elu')([outputs_of_block_source_layer[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(size_of_outputs, activation='softmax')(x)

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
        weights_path=None):
    """Build and compile one model."""

    print("Building " + model_name + " model : start")

    if source_model is not None and layer_to_transfer is not None:
        inputs = source_model.get_layer(index=0).output
    else:
        inputs = tf.keras.Input(input_shape)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=create_model(
            inputs=inputs,
            size_of_outputs=output_shape,
            layer_to_transfer=layer_to_transfer,
            source_model=source_model,
            augment_data=augment_data)
    )
    model._name = model_name

    if not trainable and weights_path is not None:
        model.load_weights(weights_path)   
        for l in model.layers:
            l.trainable = False

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("Building " + model_name + " model : done\n")

    return model
