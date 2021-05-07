from argparse import ArgumentParser
from functools import partial

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import get_augment, get_batch_augment
import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate


class BaseSupervisedDatamodule(LightningDataModule):

    def __init__(self,
                 data_dir,
                 crop_size,
                 epoch_len,
                 batch_size,
                 workers,
                 augment,
                 batch_augment,
                 train_val,
                 mixup_alpha,
                 *args,
                 **kwargs):

        super().__init__()

        self.data_dir = data_dir
        self.crop_size = crop_size
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.num_workers = workers
        self.train_val = tuple(train_val)
        self.mixup_alpha = mixup_alpha

        self.train_augment = A.Compose(
            get_augment(augment) +
            [
                A.Normalize(),
                ToTensorV2(transpose_mask=True)
            ]
        )
        self.batch_augment = get_batch_augment(batch_augment)

        self.val_augment = A.Compose([
            A.Normalize(),
            ToTensorV2(transpose_mask=True)
        ])

        self.sup_train_set = None
        self.val_set = None


    def prepare_data(self, *args, **kwargs):

        # Nothing to write on disk, data is already there, no hard
        # preprocessing necessary
        pass

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--epoch_len", type=int, default=10000)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--crop_size", type=int, default=128)
        parser.add_argument("--workers", default=8, type=int)
        parser.add_argument('--train_val', nargs=2, type=int, default=[0, 0])
        parser.add_argument('--mixup_alpha', type=float, default=0.4)
        parser.add_argument('--augment', nargs='+', type=str, default=[])
        parser.add_argument('--batch_augment', nargs='+', type=str, default=[])

        return parser

    # Following pytorch Dataloader doc, loading from a map-style dataset is
    # roughly equivalent with:
    #
    #     for indices in batch_sampler:
    #         yield collate_fn([dataset[i] for i in indices])

    def collate_labeled(self, batch, batch_augment):

        for aug in batch_augment:
            aug_batch = aug(batch=batch)
            # not necessarily self.batch_size if drop_last=False in dataloader
            batch_size = len(batch)
            idx = np.random.choice(2*batch_size, size=batch_size, replace=False)
            batch = [(batch+aug_batch)[i] for i in idx]

        return default_collate(batch)

    def wif(self, id):
        uint64_seed = torch.initial_seed()
        np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

    def train_dataloader(self):

        """
        Contrary to many standard image datasets with a lot of small images,
        remote sensing datasets come with a few big images, from which we sample
        multiple crops randomly at train time.
        """

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.collate_labeled,
                batch_augment=self.batch_augment
            ),
            sampler=sup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.wif
        )

        return sup_train_dataloader

    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.wif
        )

        return val_dataloader
