from argparse import ArgumentParser
import os
import csv

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np
import imagesize
from rasterio.windows import Window

from dl_toolbox.utils import worker_init_function
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import DigitanieDs


class DigitanieDm(LightningDataModule):

    def __init__(self,
                 #data_path,
                 splitfile_path,
                 test_fold,
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
        #self.data_path = data_path
        self.splitfile_path = splitfile_path
        self.test_fold = test_fold
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
        #parser.add_argument("--data_path", type=str)
        parser.add_argument("--splitfile_path", type=str)
        parser.add_argument("--test_fold", type=int)
        parser.add_argument("--epoch_len", type=int)
        parser.add_argument("--sup_batch_size", type=int)
        parser.add_argument("--crop_size", type=int)
        parser.add_argument("--workers", type=int)
        parser.add_argument('--img_aug', type=str)
        parser.add_argument('--batch_aug', type=str)
        parser.add_argument("--ignore_void", action='store_true')

        return parser
   
    def prepare_data(self, *args, **kwargs):

        pass

    def setup(self, stage=None):
        
        merges = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        self.labels = list(range(len(merges)))
        self.class_names = ['other',
                            'bare ground', 
                            'low vegetation',
                            'water',
                            'building',
                            'high vegetation',
                            'parking',
                            'pedestrian',
                            'road',
                            #'water',
                            'railways']
        self.label_colors = [(255,255,255), 
                             (184,141,21), 
                             (34,139,34), 
                             (0,0,238),
                             (238,118,33), 
                             (0,222,137), 
                             (118,118,118), 
                             (48,48,48), 
                             (38,38,38), 
                             #(33,203,220),
                             (112, 53,0)]
        
        train_datasets = []
        validation_datasets = []
        with open(self.splitfile_path, newline='') as splitfile:
            reader = csv.reader(splitfile)
            next(reader)
            for row in reader:
                is_val = int(row[8])==self.test_fold
                aug = 'no' if is_val else self.img_aug
                window = Window(
                    col_off=int(row[4]),
                    row_off=int(row[5]),
                    width=int(row[6]),
                    height=int(row[7])
                )
                dataset = DigitanieDs(
                    image_path=row[2],
                    label_path=row[3],
                    fixed_crops=is_val,
                    tile=window,
                    crop_size=self.crop_size,
                    img_aug=aug,
                    merge_labels=(merges, self.class_names),
                    one_hot_labels=True
                )
                if is_val:
                    validation_datasets.append(dataset)
                else:
                    train_datasets.append(dataset)
        self.train_set = ConcatDataset(train_datasets)
        self.val_set = ConcatDataset(validation_datasets)

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

