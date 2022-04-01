# Simple SSL with Lightning


It relies on:
* [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to have a common code structure with less boilerplate
* [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch) for a collection of pretrained segmentation models and state-of-the-art architectures and encoders

## Key points

* For a given dataset containing both labeled and unlabeled data, there is:
  
  * one dataset file in `torch_datasets`, containing Torch Dataset classes that define how to sample labeled and unlabeled data. 
    
  * one lightning datamodule file in `lightning_datamodules`, containing Lightning Datamodule classes that define how to build dataloaders for existing SSL methods. These classes define:
    * the supervised training/unsupervised training/validation split in `setup()`
    * the SSL train and val dataloaders necessary for the corresponding SSL method
    
* For a given SSL algo, there is one Lightning Module file in `lightning_modules`.
* Validation data in the repo should rather be understood as test data; in particular, following [[1]](#1), we should not carry out any hyperparameter search on artificially held out validation samples from the labeled pool.

## References
<a id="1">[1]</a>
Oliver, Avital, et al. "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms." Advances in Neural Information Processing Systems 31 (2018)
