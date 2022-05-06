# Toolbox for deep learning in pytorch

This repository contains pieces of code that can be combined to run machine learning algorithms. It is an ongoing work and new things may be added as I work on them.
The code is designed to ease its appropriation by others, mainly by avoiding complex heritance mechanisms and favoring one-file implementation when possible.
Libraries with significant user communities and support are also favored.

As of May 6th 2022, the code mainly provides tools to run semantic segmentation algorithms with neural networks, both supervised and semi-supervised, on remote sensing RGB images.
It is heavily based on the PytorchLightning library, which itself relies on PyTorch.
Segmentation networks are taken from the segmentation-model-pytorch library.

## Installation and test

First clone the repo. Then a good practice is to always run python codes from a project in a dedicated virtual environment : create such an environment with `virtualenv venv` (venv is the name of the virtualenv) then do `venv/bin/pip install --upgrade pip` and `venv/bin/pip install -r requirements.txt`.

To test the installation : 

* On a local machine :

* On the DeepLab :

## Usage

### Developer

Create a personal development branch (`git checkout -b <nom_branche_perso>`), push the branch on the remote (`git push -u origin <nom_branche_perso`).

### End-user

## Organisation 

### Examples of use

Complete machine learning pipelines examples can be found in /docs/examples.
As of May 6th 2022, a supervised and a semi-supervised pipeline for learning on a dataset from the AI4GEO project are given.

### Tools

Here is a short description of the pieces of code in /dl_toolbox :

* augmentations : simple implementation of data augmentation strategies based on torchvision for maximum control over their parameters.
* callbacks : callbacks implement functionalities that are not necessary for the ML pipeline to work, like visualisation tools during training. Currently callbacks are based on the PytorchLighning Callback base class.
* inference : contains tools to run inference on big raster images and compute a variety of metrics as well as visualize predictions.
* lightning datamodules : contains examples of Datamodule classes from the Lightning library for various remote sensing datasets, and both supervised and semi-supervised algorithms. Given a dataset, a datamodule mainly deals with the train/eval/test split of the data.
* lightning modules : contains Lightning code for a U-net-based semantic segmentation network acting as a strong baseline, and for the Mean Teacher algorithm for semi-supervised segmentation based on consistency regularization techniques. 
* losses : contains a reimplementation for the dice loss.
* torch_collate : in the PyTorch Dataloader class, collate functions assemble data points read from a dataset into batches. Here can be found custom examples of such collate functions to meet specific needs for batches.
* torch_datasets : custom PyTorch Dataset classes for various remote sensing datasets, with tiling capabilities.

