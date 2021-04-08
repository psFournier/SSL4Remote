from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from transforms import MergeLabels

import albumentations as A
from albumentations.pytorch import ToTensorV2
from common_utils.augmentations import get_augmentations
from datasets import MiniworldParis, MiniworldParisLabeled
from pl_datamodules import BaseSupervisedDatamodule

class MiniworldParisSup(BaseSupervisedDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        shuffled_idxs = np.random.permutation(
            len(MiniworldParis.labeled_image_paths)
        )

        val_idxs = shuffled_idxs[:self.nb_im_val]
        train_idxs = shuffled_idxs[-self.nb_im_train:]

        self.sup_train_set = MiniworldParisLabeled(
            self.data_dir, train_idxs, self.crop_size
        )

        self.val_set = MiniworldParisLabeled(
            self.data_dir, val_idxs, self.crop_size
        )