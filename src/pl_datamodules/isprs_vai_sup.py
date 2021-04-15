from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from torch_datasets import IsprsVaihingen, IsprsVaihingenLabeled
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
            IsprsVaihingen.pixels_per_class[0] / ppc for ppc in
            IsprsVaihingen.pixels_per_class
        ]
    )

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # For binary classification, all labels other than that of interest are collapsed
        self.label_merger = MergeLabels([[0], [1]])


    def setup(self, stage=None):

        nb_labeled_images = len(IsprsVaihingen.labeled_image_paths)
        labeled_idxs = list(range(nb_labeled_images))
        random.shuffle(labeled_idxs)

        nb_val_img = int(nb_labeled_images * self.prop_val)
        nb_train_img = int(nb_labeled_images * self.prop_train)
        val_idxs = labeled_idxs[:nb_val_img]
        train_idxs = labeled_idxs[-nb_train_img:]

        self.sup_train_set = IsprsVaihingenLabeled(
            data_path=self.data_dir,
            idxs=train_idxs,
            crop=self.crop_size,
            label_merger=self.label_merger,
            augmentations=self.train_augment
        )

        self.val_set = IsprsVaihingenLabeled(
            data_path=self.data_dir,
            idxs=val_idxs,
            crop=self.crop_size,
            label_merger=self.label_merger,
            augmentations=self.val_augment
        )