from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from datasets import IsprsVaihingen, IsprsVaihingenLabeled, IsprsVaihingenUnlabeled
from transforms import MergeLabels
from pl_datamodules import IsprsVaiSup

import albumentations as A
from albumentations.pytorch import ToTensorV2

class IsprsVaiSemisup(IsprsVaiSup):

    def __init__(self, nb_im_unsup_train, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.nb_im_unsup_train = nb_im_unsup_train

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--nb_im_unsup_train", type=int, default=0)

        return parser

    def setup(self, stage=None):

        shuffled_idxs = list(
            np.random.permutation(
                len(IsprsVaihingen.labeled_image_paths)
            )
        )

        val_idxs = shuffled_idxs[:self.nb_im_val]
        train_idxs = shuffled_idxs[-self.nb_im_train:]

        self.sup_train_set = IsprsVaihingenLabeled(
            self.data_dir, train_idxs, self.crop_size
        )

        self.val_set = IsprsVaihingenLabeled(
            self.data_dir, val_idxs, self.crop_size
        )

        # ...but each non validation labeled image can be used without its
        # label for unsupervised training
        unlabeled_idxs = list(range(len(IsprsVaihingen.unlabeled_image_paths)))
        all_unsup_train_idxs = shuffled_idxs[self.nb_im_val:] + \
                              unlabeled_idxs
        unsup_train_idxs = all_unsup_train_idxs[:self.nb_im_unsup_train]
        self.unsup_train_set = IsprsVaihingenUnlabeled(
            self.data_dir,
            unsup_train_idxs,
            self.crop_size
        )

    def collate_unlabeled(self, batch):

        transformed_batch = [
            self.augmentations(
                image=image
            )
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
            collate_fn=self.collate_labeled,
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
            collate_fn=self.collate_unlabeled,
            sampler=unsup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader,
        }

        return train_dataloaders
