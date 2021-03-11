from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

from datasets import IsprsVaihingen, IsprsVaihingenLabeled, IsprsVaihingenUnlabeled


class RotEquivarianceIsprsVaihingen(LightningDataModule):
    def __init__(
        self,
        data_path,
        crop_size,
        nb_pass_per_epoch,
        batch_size,
        sup_train_transforms,
        val_transforms,
        unsup_train_transforms,
    ):

        super().__init__()

        self.data_path = data_path
        self.crop_size = crop_size
        self.nb_pass_per_epoch = nb_pass_per_epoch
        self.batch_size = batch_size
        self.sup_train_transforms = sup_train_transforms
        self.val_transforms = val_transforms
        self.unsup_train_transforms = unsup_train_transforms

    def prepare_data(self, *args, **kwargs):

        # Nothing to write on disk, data is already there, no hard
        # preprocessing necessary
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--nb_pass_per_epoch", type=int, default=1)
        parser.add_argument(
            "--data_dir", type=str, default="/home/pierre/Documents/ONERA/ai4geo/"
        )
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--crop_size", type=int, default=128)

        return parser

    def setup(self, stage=None):

        labeled_idxs = IsprsVaihingen.labeled_idxs
        np.random.shuffle(labeled_idxs)

        # Here we use very few labeled images for training (2)...
        val_idxs = labeled_idxs[:7]
        train_idxs = labeled_idxs[14:]

        self.sup_train_set = IsprsVaihingenLabeled(
            self.data_path, train_idxs, self.crop_size, self.sup_train_transforms
        )

        self.val_set = IsprsVaihingenLabeled(
            self.data_path, val_idxs, self.crop_size, self.val_transforms
        )

        # ...but each non validation labeled image is used without its label for
        # unsupervised training
        unlabeled_idxs = IsprsVaihingen.unlabeled_idxs
        unsup_train_idxs = labeled_idxs[7:] + unlabeled_idxs
        self.unsup_train_set = IsprsVaihingenUnlabeled(
            self.data_path,
            unsup_train_idxs,
            self.crop_size,
            self.unsup_train_transforms,
        )

    def train_dataloader(self):

        """
        Contrary to many standard image datasets with a lot of small images,
        remote sensing datasets like ISPRS Vaihingen come with a few big images.
        Thus dataset classes get_item functions provide only a crop of the image.
        For an epoch to actually span around the entirety of the dataset, we thus
        need to sample mutliple times randomly from each big image. Hence the need
        for RandomSamplers and for the nb_pass_per_epoch parameter, otherwise the
        agent would see a single crop from each image during an epoch.
        """

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=self.nb_pass_per_epoch * len(self.sup_train_set),
        )

        # num_workers should be the number of cpus on the machine.
        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            sampler=sup_train_sampler,
            num_workers=2,
            pin_memory=True,
        )

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.nb_pass_per_epoch * len(self.unsup_train_set),
        )

        # num_workers should be the number of cpus on the machine.
        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.batch_size,
            sampler=unsup_train_sampler,
            num_workers=2,
            pin_memory=True,
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader,
        }

        return train_dataloaders

    def val_dataloader(self):

        num_samples = self.nb_pass_per_epoch * len(self.val_set)
        val_sampler = RandomSampler(
            data_source=self.val_set, replacement=True, num_samples=num_samples
        )

        # num_workers should be the number of cpus on the machine.
        val_dataloader = DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
        )

        return val_dataloader
