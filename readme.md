# QUANTA: QUANtitative TrAnsferability

## Submitted abstract to NIPS 2020
> The computationally expansive nature of Deep Neural Networks, along with their
significant hunger for labeled data, can impair the overall performance of these
models. Among other techniques, this challenge can be tackled by Transfer
Learning, which consist in re-using the knowledge previously learned by a model:
this method is widely used and has proven effective in enhancing the performance
of models in low resources contexts. However, there are relatively few 
contributions regarding the actual transferability of features in a deep 
learning model. This paper presents QUANTA (QUANtitative TrAnsferability), 
a method for quantifying the transferability of features of a given model. 
A QUANTA is a 2-parameters layer added in a target model at the level at which 
one wants to study the transferability of the corresponding layer level in a 
source model. Data from the target domain being fed to both the source and the 
target models, the parameters of the QUANTA layer are trained in such a way 
that a mutually exclusive quantification occurs between the source model 
(trained and frozen) and the (trainable) target model. The proposed approach 
is evaluated on a set of experiments on a visual recognition task using 
Convolutional Neural Networks. The results show that QUANTA are a promising 
tool for quantifying the transferability of features of a source model, as well 
as new way of assessing the quality of a transfer.

## Pre-requisite
The code was developped using [Python 3](https://www.python.org/downloads/) and [Keras over Tensorflow 2](https://www.tensorflow.org/install)

## What's there?
```
.
├── CREDITS.md
├── LICENSE.txt
├── SourceModel.h5
├── README.md
├── models.py
├── quanta.py
├── QuantaCustomCallback.py
├── QuantaLayer.py
```

## Running the code
Use the bash script ``start_expe.sh`` to start an experiment:

``./start_expe -o|--outdir=outputdir -r|--repeat=30 -l|--layer=5 -t|--targetTask=cifar10``

Options:
* ``-o|--outdir=`` the output directory where the results of the experiements are written
* ``-r|--repeat=`` for executing x runs of 60 epochs
* ``-l|--layer=`` the layer whose transferability is assessed (0 to 6)
* ``-t|--targetTask=`` 'cifar10' or 'cifar100'
* ``--seed=`` for setting the seed to manage determinism

## Exported data
After a run is complete, data are exported in the specified output directory, which
has the following structure:
```
.
├── specified_output_dir
│   ├── 0.txt
│   ├── 0_raw.txt
│   ├── ...
│   ├── testing_metrics_of_target_0.txt
│   ├── ...
│   └── training_metrics_of_target_0.txt
│   └── ...
├── ...
├── README.md
└── ...
```
where ``0.txt`` (``r.txt``) contains the evolution of the measured transferability
over the 60 epochs of run number 0 (run number ``r``). Files suffixed ``raw``
contain the values of ``v`` (that is, before being transformed through a softmax).
The file ``training_metrics_of_target_0.txt`` (resp. ``training_metrics_of_target_r.txt``)
contains all the exported target model training metrics over the 60 epochs of run 0 (resp. ``r``).
The file ``testing_metrics_of_target_0.txt`` (resp. ``testing_metrics_of_target_r.txt``)
contains all the exported target model testing metrics over the 60 epochs of run 0 (resp. ``r``).

## Licence
This project is licensed under the Mozilla Public Licence 2.0. See the ``LICENSE.txt``
for details.