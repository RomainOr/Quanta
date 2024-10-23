# QUANTA: QUANtitative TrAnsferability

## Paper in the proceedings of the 25th International Conference KES2021

> The computationally expensive nature of Deep Neural Networks, along with their significant hunger for labeled data, can impair the overall performance of these models. Among other techniques, this challenge can be tackled by Transfer Learning, which consists in re-using the knowledge previously learned by a model: this method is widely used and has proven effective in enhancing the performance of models in low resources contexts. However, there are relatively few contributions regarding the actual transferability of features in a deep learning model. This paper presents QUANTA (QUANtitative TrAnsferability), a method for quantifying the transferability of features of a given model. A QUANTA is a 2-parameters layer added in a target model at the level at which one wants to study the transferability of the corresponding layer in a source model. Data from the target domain being fed to both the source and the target models, the parameters of the QUANTA layer are trained in such a way that a mutually exclusive quantification occurs between the source model (trained and frozen) and the (trainable) target model. The proposed approach is evaluated on a set of experiments on a visual recognition task using Convolutional Neural Networks. The results show that QUANTA is a promising tool for quantifying the transferability of features of a source model, as well as a new way of assessing the quality of a transfer.

## Pre-requisite

The code was developped using [Python 3](https://www.python.org/downloads/) and [Keras over Tensorflow 2](https://www.tensorflow.org/install).

A source model should already be trained before using quanta and its model has to be stored in a file named ``SourceModel.keras`` at the root of the quanta evaluation tool.

Be also careful of the architectures of your source model and your target model to use quanta.

## Running the code

``usage: quanta.py [-h] [-o OUTPUT_DIR] -s {cifar10} -sw SOURCE_WEIGHTS -t {cifar10,cifar100} -l LAYER_TO_TRANSFER [-r NB_OF_RUNS] [--nb_of_target_epochs NB_OF_TARGET_EPOCHS] [--nb_of_target_samples NB_OF_TARGET_SAMPLES] [--seed SEED] [--augment_data] [--train_from_previous_training TRAIN_FROM_PREVIOUS_TRAINING]``

Options:
* ``-h|--help`` Show help message and usage.
* ``-o|--outdir`` Precise the output directory where the results of the experiements are written. If the output directory does not exist, it will be created by the program. Default value is **./expe**.
* ``-s|--source_task`` Define the source task of the pre-trained source model. Have to be 'cifar10'.
* ``-sw|--source_weights`` Define the path of the weights of the pre-trained source model. Required and has to be a h5 file.
* ``-t|--target_task`` Define the target task of the target model to train. Have to be 'cifar10' or 'cifar100'.
* ``-l|--layer_to_transfer`` Define the layer whose transferability is assessed. Be carefull to respect your models. If this argument is set to '-1' or a negative value, then all source layers that can be transfered are transfered at the same time.
* ``-r|--nb_of_runs`` Define the number of runs to experiment a transfer with quanta. Have to be stricly positive. Default value is **1**.
* ``--nb_of_target_epochs`` Define the number of epochs to train a target model. Have to be stricly positive. Default value is **1**.
* ``--nb_of_target_samples`` Define the number of samples to train and test a target model. Have to be stricly positive. This value is also used to compare performance between source and target models.
* ``--seed`` Define the seed to manage determinism. If positive, the value is used to get determinism, otherwise no seed is set.
* ``--augment_data`` If this flag is given, the target model will be trained with augmented data.
* ``--train_from_previous_training`` If the argument following this option is valid, the target model will be trained from a previous saved model located by this argument.
* ``--show_only_build`` If this flag is given, the models will only be compiled and their tf summary will be displayed.

Examples :
* ``python quanta.py -h``
* ``python quanta.py -o ./expe -s cifar10 -sw SourceModel.weights.h5 -t cifar10 -l 0 -r 1 --seed 42 --nb_of_target_samples 320``
* ``python quanta.py -s cifar10 -sw SourceModel.weights.h5 -t cifar10 -l 0 -r 1 --seed 42 --nb_of_target_samples 320 --train_from_previous_training ./expe/target_r_0_l_0.keras``

## Exported data

After a run is complete, data are exported in the specified output directory {OUTPUT_DIR} , which
has the following structure:
```
.
├── specified_output_dir
│   └── {source_task}_to_{target_task}
│   │   ├── ...
│   │   ├── source_r_{number_of_run}_l_{layer_to_transfer}.keras
│   │   ├── ...
│   │   ├── target_r_{number_of_run}_l_{layer_to_transfer}.keras
│   │   ├── ...
│   │   ├── quantas_r_{number_of_run}_l_{layer_to_transfer}.jsonl
│   │   ├── ...
│   │   ├── testing_metrics_of_source_r_{number_of_run}_l_{layer_to_transfer}.jsonl
│   │   ├── ...
│   │   ├── testing_metrics_of_target_r_{number_of_run}_l_{layer_to_transfer}.jsonl
│   │   ├── ...
│   │   ├── training_metrics_of_target_r_{number_of_run}_l_{layer_to_transfer}.jsonl
│   │   └── ...
```

where, given a {LAYER_TO_TRANSFER} and a {run} out of {NB_OF_RUNS}:
* ``source_r_{run}_l_{layer_to_transfer}.keras`` is the keras model of source.
* ``target_r_{run}_l_{layer_to_transfer}.keras`` is the keras model of target.
* ``quantas_r_{number_of_run}_l_{layer_to_transfer}.jsonl`` collects all quanta values within the target model.
* ``testing_metrics_of_source_r_{run}_l_{layer_to_transfer}.jsonl`` collects all testing metrics of the source model.
* ``testing_metrics_of_target_r_{run}_l_{layer_to_transfer}.jsonl`` collects all testing metrics of the target model.
* ``training_metrics_of_target_r_{run}_l_{layer_to_transfer}.jsonl`` collects all training metrics of the target model.

When tensorflow displays the measured metrics, be carefull about the fact that it averages the metrics over the batches used to train or test a model.

## Determinism by seeding

Be careful that the states of the PRNG are different, and so the results are, when you reload a model to continue a training from when you just continue your training without exiting the program.

## License
This project is licensed under the Mozilla Public Licence 2.0. See the ``LICENSE.txt`` for details.
