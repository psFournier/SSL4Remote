from argparse import ArgumentParser
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np

from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import DigitanieDs


class DigitanieDm(LightningDataModule):

    def __init__(self,
                 data_path,
                 crop_size,
                 epoch_len,
                 sup_batch_size,
                 workers,
                 img_aug,
                 batch_aug,
                 ignore_void,
                 *args,
                 **kwargs):

        super().__init__()
        self.data_path = data_path
        self.class_names = [label[2] for label in DigitanieDs.labels_desc]
        self.crop_size = crop_size
        self.epoch_len = epoch_len
        self.sup_batch_size = sup_batch_size
        self.num_workers = workers
        self.ignore_void = ignore_void
        self.img_aug = img_aug
        self.batch_aug = batch_aug

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--epoch_len", type=int, default=10000)
        parser.add_argument("--sup_batch_size", type=int, default=16)
        parser.add_argument("--crop_size", type=int, default=128)
        parser.add_argument("--workers", default=6, type=int)
        parser.add_argument('--img_aug', type=str, default='no')
        parser.add_argument('--batch_aug', type=str, default='no')
        parser.add_argument("--ignore_void", action='store_true')

        return parser
   
    def prepare_data(self, *args, **kwargs):

        pass

    def setup(self, stage=None):
        
        tile_names = ['arenes', 'bagatelle', 'cepiere', 'empalot', 'lardenne', 'minimes', 'mirail', 'montaudran']
        datasets = [DigitanieDs(
            image_path=os.path.join(self.data_path, 'Toulouse', f'tlse_{tile}_img_c.tif'),
            label_path=os.path.join(self.data_path, 'Toulouse', f'tlse_{tile}_c.tif'),
            fixed_crops=False,
            crop_size=128,
            img_aug=self.img_aug) for tile in tile_names
            ]
        self.train_set = ConcatDataset(datasets)

        tile_names = ['ramier', 'zenith']
        datasets = [DigitanieDs(
            image_path=os.path.join(self.data_path, 'Toulouse', f'tlse_{tile}_img_c.tif'),
            label_path=os.path.join(self.data_path, 'Toulouse', f'tlse_{tile}_c.tif'),
            fixed_crops=True,
            crop_size=128,
            img_aug=None) for tile in tile_names
        ]
        self.val_set = ConcatDataset(datasets)
    
    def train_dataloader(self):

        train_sampler = RandomSampler(
            data_source=self.train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        train_dataloader = DataLoader(
            dataset=self.train_set,
            batch_size=self.sup_batch_size,
            collate_fn=CustomCollate(self.batch_aug),
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )

        return train_dataloader

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
    
    def label_to_rgb(self, labels):

        rgb_label = np.zeros(shape=(*labels.shape, 3), dtype=float)
        for val, color, _ in DigitanieDs.labels_desc:
            mask = np.array(labels == val)
            rgb_label[mask] = np.array(color)
        rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

        return rgb_label

