from argparse import ArgumentParser

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate
import torch

from utils import worker_init_function
from torch_collate import CustomCollate
from augmentations import get_transforms

class BaseSupervisedDatamodule(LightningDataModule):

    def __init__(self,
                 crop_size,
                 epoch_len,
                 sup_batch_size,
                 workers,
                 train_val,
                 train_idxs,
                 val_idxs,
                 img_aug,
                 batch_aug,
                 ignore_void=False,
                 *args,
                 **kwargs):

        super().__init__()

        self.crop_size = crop_size
        self.epoch_len = epoch_len
        self.sup_batch_size = sup_batch_size
        self.num_workers = workers
        self.train_val = tuple(train_val)
        self.train_idxs = list(train_idxs)
        self.val_idxs = list(val_idxs)
        self.sup_train_set = None
        self.val_set = None
        self.ignore_void = ignore_void
        self.img_aug = img_aug
        self.batch_aug = batch_aug

    def prepare_data(self, *args, **kwargs):

        # Nothing to write on disk, data is already there, no hard
        # preprocessing necessary
        pass

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--epoch_len", type=int, default=10000)
        parser.add_argument("--sup_batch_size", type=int, default=16)
        parser.add_argument("--crop_size", type=int, default=128)
        parser.add_argument("--workers", default=6, type=int)
        parser.add_argument('--train_val', nargs=2, type=int, default=(0, 0))
        parser.add_argument('--train_idxs', nargs='+', type=int, default=[])
        parser.add_argument('--val_idxs', nargs='+', type=int, default=[])
        parser.add_argument('--img_aug', type=str, default='no')
        # parser.add_argument('--val_dataset_transforms_strat', type=str, default='no')
        parser.add_argument('--batch_aug', type=str, default='no')
        parser.add_argument("--ignore_void", action='store_true')

        return parser

    def train_dataloader(self):

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.sup_batch_size,
            collate_fn=CustomCollate(self.batch_aug),
            sampler=sup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )

        return sup_train_dataloader

    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(),
            batch_size=self.sup_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )

        return val_dataloader
