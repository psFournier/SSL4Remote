from argparse import ArgumentParser
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from transforms import MergeLabels

import albumentations as A
from albumentations.pytorch import ToTensorV2
from common_utils.augmentations import get_augmentations
from pl_datamodules import BaseSupervisedDatamodule

class BaseSemisupDatamodule(BaseSupervisedDatamodule):

    def __init__(self, nb_im_unsup_train, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.nb_im_unsup_train = nb_im_unsup_train
        self.unsup_train_set = None

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--nb_im_unsup_train", type=int, default=0)

        return parser

    def collate_unlabeled(self, batch, augment):

        transformed_batch = [
            augment(image=image)
            for image in batch
        ]
        batch = [(elem["image"]) for elem in transformed_batch]

        return default_collate(batch)

    def train_dataloader(self):

        """
        See the supervised dataloader for comments on the need for samplers.
        The semi supervised loader consists in two loaders for labeled and
        unlabeled data.
        """

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=int(self.nb_pass_per_epoch * len(self.sup_train_set)),
        )

        # num_workers should be the number of cpus on the machine.
        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.collate_labeled,
                augment=self.train_augment
            ),
            sampler=sup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=int(self.nb_pass_per_epoch * len(self.unsup_train_set)),
        )
        # num_workers should be the number of cpus on the machine.
        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.collate_unlabeled,
                augment=self.train_augment
            ),
            sampler=unsup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader,
        }

        return train_dataloaders