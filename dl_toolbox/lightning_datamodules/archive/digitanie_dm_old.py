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
        
        merges = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        self.labels = list(range(len(merges)))
        self.class_names = ['void','bareland', 'low vegetation', 'water', 'building',
                       'high vegetation', 'parking',  'pedestrian', 'road', 'pool', 'railway']
        self.label_colors = [(255,255,255), (184,141,21), (34,139,34), (0,0,238),
                        (238,118,33), (0,222,137), (118,118,118), (48,48,48), (38,38,38), (33,203,220),
                        (112, 53,0)]

        tlse_train = ['bagatelle', 'cepiere', 'lardenne', 'minimes', 'mirail',
                      'montaudran', 'zenith', 'ramier']
        other_train = [f'tuile_{i}' for i in range(1, 9)]
        
        datasets = [
            DigitanieDs(
                image_path=os.path.join(
                    self.data_path, 
                    city, 
                    city_lower+f'_{tile}_img_c.tif'
                ),
                label_path=os.path.join(
                    self.data_path, 
                    city, 
                    city_lower+f'_{tile}_c.tif'
                ),
                fixed_crops=False,
                crop_size=self.crop_size,
                img_aug=self.img_aug,
                merge_labels=(merges, self.class_names),
                one_hot_labels=True
            ) for city, city_lower, tile_names in [
                ('Toulouse', 'tlse', tlse_train),
                ('Paris', 'paris', other_train), 
                ('Biarritz', 'biarritz', other_train), 
                ('Strasbourg', 'strasbourg', other_train)
            ] for tile in tile_names
        ]
        self.train_set = ConcatDataset(datasets)

        tlse_test = ['empalot', 'arenes']
        other_test = [f'tuile_{i}' for i in range(9, 11)]

        datasets = [
            DigitanieDs(
                image_path=os.path.join(
                    self.data_path, 
                    city, 
                    city_lower+f'_{tile}_img_c.tif'
                ),
                label_path=os.path.join(
                    self.data_path, 
                    city, 
                    city_lower+f'_{tile}_c.tif'
                ),
                fixed_crops=True,
                crop_size=self.crop_size,
                img_aug=self.img_aug,
                merge_labels=(merges, self.class_names),
                one_hot_labels=True
            ) for city, city_lower, tile_names in [
                ('Toulouse', 'tlse', tlse_test),
                ('Paris', 'paris', other_test), 
                ('Biarritz', 'biarritz', other_test), 
                ('Strasbourg', 'strasbourg', other_test)
            ] for tile in tile_names
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
        for val, color in zip(self.labels, self.label_colors):
            mask = np.array(labels == val)
            rgb_label[mask] = np.array(color)
        rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

        return rgb_label

class DigitanieSemisupDm(DigitanieDm):

    def __init__(
        self,
        unsup_batch_size,
        unsup_crop_size,
        *args,
        **kwargs
    ):
        
        super().__init__(*args, **kwargs)
        self.unsup_batch_size = unsup_batch_size
        self.unsup_crop_size = unsup_crop_size

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--unsup_batch_size", type=int, default=16)
        parser.add_argument("--unsup_crop_size", type=int, default=160)
        return parser

    def setup(self, stage=None):

        super().setup()
        self.unsup_train_set = DigitanieDs(
            image_path=os.path.join(self.data_path, 'Toulouse',
                                    'normalized_mergedTO.tif'),
            fixed_crops=False,
            crop_size=self.unsup_crop_size,
            img_aug=self.img_aug
        )

    def train_dataloader(self):

        train_dataloader = super().train_dataloader()
        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.unsup_batch_size,
            sampler=unsup_train_sampler,
            collate_fn=CustomCollate(),
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )

        train_dataloaders = {
            "sup": train_dataloader,
            "unsup": unsup_train_dataloader
        }

        return train_dataloaders 

