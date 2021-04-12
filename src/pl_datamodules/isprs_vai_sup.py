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

class IsprsVaiSup(BaseSupervisedDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        nb_labeled_images = len(IsprsVaihingen.labeled_image_paths)
        labeled_idxs = list(range(nb_labeled_images))
        random.shuffle(labeled_idxs)

        nb_val_img = int(nb_labeled_images * self.prop_val)
        nb_train_img = int(nb_labeled_images * self.prop_train)
        val_idxs = labeled_idxs[:nb_val_img]
        train_idxs = labeled_idxs[-nb_train_img:]

        self.sup_train_set = IsprsVaihingenLabeled(
            self.data_dir, train_idxs, self.crop_size
        )

        self.val_set = IsprsVaihingenLabeled(
            self.data_dir, val_idxs, self.crop_size
        )