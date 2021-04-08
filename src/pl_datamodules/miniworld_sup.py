from argparse import ArgumentParser
from functools import partial
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate

from transforms import MergeLabels

import albumentations as A
from albumentations.pytorch import ToTensorV2
from common_utils.augmentations import get_augmentations
from datasets import MiniworldParis, MiniworldParisLabeled
from pl_datamodules import BaseClassSupervised

class MiniworldSup(BaseClassSupervised):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        sup_train_datasets = []
        val_datasets = []
        city_classes = [
            getattr('datasets', name) for name in [
                'MiniworldParisLabeled',
                'MiniworldArlingtonLabeled'
            ]
        ]

        for city_class in city_classes:

            shuffled_idxs = np.random.permutation(
                len(city_class.labeled_image_paths)
            )

            val_idxs = shuffled_idxs[:self.nb_im_val]
            train_idxs = shuffled_idxs[-self.nb_im_train:]

            sup_train_datasets.append(
                city_class(self.data_dir, train_idxs, self.crop_size)
            )

            val_datasets.append(
                city_class(self.data_dir, val_idxs, self.crop_size)
            )

        self.sup_train_set = ConcatDataset(sup_train_datasets)
        self.val_set = ConcatDataset(val_datasets)