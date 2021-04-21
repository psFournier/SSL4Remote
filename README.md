# Simple SSL with Lightning

This repo contains an easy to reuse implementation of a somewhat trivial approach
to semi supervised learning. 
It should also be used as a guide (to discuss) on how to implement future SSL methods.


It relies on:
* [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to have a common code structure with less boilerplate
* [Albumentations](https://github.com/albumentations-team/albumentations) for image augmentations
* [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) for a collection of pretrained segmentation models and state-of-the-art architectures and encoders

## Key points

* For a given dataset containing both labeled and unlabeled data, there is:
  
  * one dataset file in `src/datasets`, containing Torch Dataset classes that define how to sample labeled and unlabeled data. These classes are otherwise agnostic to the future processing methods of the data.
    
  * one lightning datamodule file in `src/datamodules`, containing Lightning Datamodule classes that define how to build dataloaders for existing SSL methods. These classes define:
    * the supervised training/unsupervised training/validation split in `setup()`
    * the SSL train and val dataloaders necessary for the corresponding SSL method

> **_NOTE:_** For now I have not decided whether there should be one datamodule per method or one per pair dataset/method. What is gained in genericity is lost in ease of reuse.
    
* For a given SSL algo, there is one Lightning Module file in `src/pl_modules`. The "research code" most probably lies in the `training_step()` method, as it should with Lightning modules.
* The `src/networks` is only used if we need networks that are not in segmentation-models-pytorch.
* Validation data in the repo should rather be understood as test data; in particular, following [[1]](#1), we should not carry out any hyperparameter search on artificially held out validation samples from the labeled pool.

## To detail

Parameters of the trainer that do not appear (including how to deal with 
different sup/unsup datasets)

## Launching experiments

### In a virtualenv on the Hal cluster 

To detail

### On the VRE

To detail

## References
<a id="1">[1]</a>
Oliver, Avital, et al. "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms." Advances in Neural Information Processing Systems 31 (2018)
