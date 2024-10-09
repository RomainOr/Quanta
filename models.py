"""Collection of functions to build models used with tensorflow."""

import tensorflow as tf

from quanta_layer import QuantaLayer


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

def create_model(inputs, size_of_outputs, layer_to_tranfer, source_model=None, augment_data=False):
    """Build a model made from convolutionnal blocks from *build_block* function."""

    outputs_of_block_source_layer = []
    if source_model is not None:
        outputs_of_block_source_layer = [
            layer.output
            for layer in source_model.layers
            if type(layer).__name__ in ('Conv2D', 'Dense')]

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    if source_model is not None and augment_data:
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

    if (outputs_of_block_source_layer != [] and (layer_to_tranfer in (0, 1))):
        x = build_block(x, 64, layer_to_tranfer, [
            outputs_of_block_source_layer[0], outputs_of_block_source_layer[1]])
    else:
        x = build_block(x, 64)

    if (outputs_of_block_source_layer != [] and (layer_to_tranfer in (2, 3))):
        x = build_block(x, 128, layer_to_tranfer, [
            outputs_of_block_source_layer[2], outputs_of_block_source_layer[3]])
    else:
        x = build_block(x, 128)

    if (outputs_of_block_source_layer != [] and (layer_to_tranfer in (4, 5))):
        x = build_block(x, 512, layer_to_tranfer, [
            outputs_of_block_source_layer[4], outputs_of_block_source_layer[5]])
    x = build_block(x, 512)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2048)(x)
    if (outputs_of_block_source_layer != [] and layer_to_tranfer == 6):
        x = QuantaLayer('elu')([outputs_of_block_source_layer[6], x])
    else:
        x = tf.keras.layers.Activation('elu')(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(size_of_outputs, activation='softmax')(x)

    return x


#################################################
####### Configure the models for training #######
#################################################

def compile_models(
        input_shape, output_shape, layer_to_tranfer, optimizer, loss, metrics, augment_data):
    """Compile all models needed for a transfer."""

    print("Building source and target models : start")

    input_source = tf.keras.Input(input_shape, name='input_source')
    input_target = tf.keras.Input(input_shape, name='input_target')

    source_model = tf.keras.Model(
        inputs=input_source,
        outputs=create_model(
            inputs=input_source,
            size_of_outputs=10,
            layer_to_tranfer=layer_to_tranfer)
    )
    source_model._name = "source"

    copy_of_source_model = tf.keras.Model(
        inputs=input_target,
        outputs=create_model(
            inputs=input_target,
            size_of_outputs=10,
            layer_to_tranfer=layer_to_tranfer)
    )
    copy_of_source_model._name = "source_copy"
    copy_of_source_model.load_weights('./SourceModel.weights.h5')
    for l in copy_of_source_model.layers:
        l.trainable = False

    target_model = tf.keras.Model(
        inputs=input_target,
        outputs=create_model(
            inputs=input_target,
            size_of_outputs=output_shape,
            layer_to_tranfer=layer_to_tranfer,
            source_model=copy_of_source_model,
            augment_data=augment_data)
    )
    target_model._name = "target"

    source_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    target_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("Building source and target models : done\n")

    return source_model, copy_of_source_model, target_model
