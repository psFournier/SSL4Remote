from argparse import ArgumentParser
from functools import partial

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.augmentations import get_augment
import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate
from utils import Mixup


class BaseSupervisedDatamodule(LightningDataModule):

    def __init__(self,
                 data_dir,
                 crop_size,
                 epoch_len,
                 batch_size,
                 workers,
                 augment,
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
        self.val_augment = A.Compose([
            A.Normalize(),
            ToTensorV2(transpose_mask=True)
        ])
        self.mixup = Mixup(alpha=0.4)

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
        # parser.add_argument('--augmentations', type=str, default='no',
        #                     help="Which augmentation strategy to use. See utils.augmentations.py")
        parser.add_argument('--train_val', nargs=2, type=int, default=[0, 0])
        parser.add_argument('--mixup_alpha', type=float, default=0.4)
        parser.add_argument('--augment', nargs='+', type=str, default=[])

        return parser

    # Following pytorch Dataloader doc, loading from a map-style dataset is
    # roughly equivalent with:
    #
    #     for indices in batch_sampler:
    #         yield collate_fn([dataset[i] for i in indices])

    def collate_labeled(self, batch):

        # We apply transforms here because transforms are method-dependent
        # while the dataset class should be method independent.
        # transformed_batch = [
        #     augment(
        #         image=image,
        #         mask=self.label_merger(ground_truth)
        #     )
        #     for image,ground_truth in batch
        # ]
        # batch = [(elem["image"], elem["mask"]) for elem in transformed_batch]

        mixed_batch = self.mixup(batch=batch)
        idx = np.random.choice(2*self.batch_size, size=self.batch_size, replace=False)
        rand_mixed_batch = [(batch+mixed_batch)[i] for i in idx]

        return default_collate(rand_mixed_batch)

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
                self.collate_labeled
                # augment=self.train_augment
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
