from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from torch_datasets import IsprsV, IsprsVLabeled
from transforms import MergeLabels

import albumentations as A
from albumentations.pytorch import ToTensorV2
from common_utils.augmentations import get_augmentations
from pl_datamodules import BaseSupervisedDatamodule
import random
from torch import tensor

class IsprsVaiSup(BaseSupervisedDatamodule):

    class_weights = tensor(
        [
            IsprsV.pixels_per_class[0] / ppc for ppc in
            IsprsV.pixels_per_class
        ]
    )

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # For binary classification, all labels other than that of interest are collapsed
        self.label_merger = MergeLabels([[0], [1]])


    def setup(self, stage=None):

        nb_labeled_images = IsprsV.nb_labeled_images
        labeled_idxs = list(range(nb_labeled_images))

        nb_val_img = int(nb_labeled_images * 0.2)
        nb_train_img = int(nb_labeled_images * 0.8)
        val_idxs = labeled_idxs[:nb_val_img]
        train_idxs = labeled_idxs[-nb_train_img:]

        self.sup_train_set = IsprsVLabeled(
            data_path=self.data_dir,
            idxs=train_idxs,
            crop=self.crop_size,
            label_merger=self.label_merger,
            augmentations=self.train_augment
        )

        self.val_set = IsprsVLabeled(
            data_path=self.data_dir,
            idxs=val_idxs,
            crop=self.crop_size,
            label_merger=self.label_merger,
            augmentations=self.val_augment
        )