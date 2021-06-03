from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

import torch
import numpy as np
from utils import get_image_level_aug, get_batch_level_aug
from augmentations import Compose


class BaseSupervisedDatamodule(LightningDataModule):

    def __init__(self,
                 data_dir,
                 crop_size,
                 epoch_len,
                 batch_size,
                 workers,
                 img_aug,
                 aug_prob,
                 batch_aug,
                 train_val,
                 *args,
                 **kwargs):

        super().__init__()

        self.data_dir = data_dir
        self.crop_size = crop_size
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.num_workers = workers
        self.train_val = tuple(train_val)
        self.sup_train_set = None
        self.val_set = None
        self.img_aug = Compose(get_image_level_aug(names=img_aug, p=aug_prob))
        self.batch_aug = get_batch_level_aug(name=batch_aug)

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
        parser.add_argument('--train_val', nargs=2, type=int)
        parser.add_argument('--img_aug', nargs='+', type=str, default=[])
        parser.add_argument('--aug_prob', type=float, default=0.7)
        parser.add_argument('--batch_aug', type=str, default='no')

        return parser

    def wif(self, id):
        uint64_seed = torch.initial_seed()
        np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

    def collate_and_aug(self, batch):

        batch = default_collate(batch)

        try:
            img, label = batch
        except:
            img, label = batch, None

        batch = self.img_aug(img=img, label=label)
        batch = self.batch_aug(*batch)
        if len(batch) < 3:
            s = batch[0].size()
            batch = (*batch, torch.ones(size=(s[0], s[2], s[3])))

        return batch

    def train_dataloader(self):

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_and_aug,
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
