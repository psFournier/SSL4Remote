from argparse import ArgumentParser
import os
from functools import partial
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
from dl_toolbox.torch_datasets.utils import *


def build_datasets_from_csv(splitfile, test_fold, img_aug, data_path, crop_size, merges, class_names):
    validation_datasets, train_datasets = [], []
    reader = csv.reader(splitfile)
    
    m, M = np.array(DigitanieDs.DATASET_DESC['min'][:3]), np.array(DigitanieDs.DATASET_DESC['max'][:3])
    next(reader)
    for row in reader:
        is_val = int(row[8]) in test_fold
        aug = 'no' if is_val else img_aug
        window = Window(
            col_off=int(row[4]),
            row_off=int(row[5]),
            width=int(row[6]),
            height=int(row[7])
        )
        dataset = DigitanieDs(
            image_path=os.path.join(data_path, row[2]),
            label_path=os.path.join(data_path, row[3]),
            fixed_crops=is_val,
            tile=window,
            crop_size=crop_size,
            crop_step=crop_size,
            read_window_fn=partial(
                read_window_from_big_raster, 
                raster_path=os.path.join(data_path, row[9])
            ),
            norm_fn=partial(
                minmax,
                m=m,
                M=M
            ),
            img_aug=aug,
            merge_labels=(merges, class_names),
            one_hot_labels=True
        )
        if is_val:
            validation_datasets.append(dataset)
        else:
            train_datasets.append(dataset)
    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(validation_datasets)
    return train_set, val_set


class DigitanieDm(LightningDataModule):

    def __init__(self,
                 data_path,
                 splitfile_path,
                 test_fold,
                 crop_size,
                 epoch_len,
                 sup_batch_size,
                 workers,
                 img_aug,
                 batch_aug,
                 *args,
                 **kwargs):

        super().__init__()
        self.data_path = data_path
        self.splitfile_path = splitfile_path
        self.test_fold = test_fold
        self.crop_size = crop_size
        self.epoch_len = epoch_len
        self.sup_batch_size = sup_batch_size
        self.num_workers = workers
        self.img_aug = img_aug
        self.batch_aug = batch_aug

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--splitfile_path", type=str)
        parser.add_argument("--test_fold", nargs='+', type=int)
        parser.add_argument("--epoch_len", type=int)
        parser.add_argument("--sup_batch_size", type=int)
        parser.add_argument("--crop_size", type=int)
        parser.add_argument("--workers", type=int)
        parser.add_argument('--img_aug', type=str)
        parser.add_argument('--batch_aug', type=str)

        return parser
   
    def prepare_data(self, *args, **kwargs):

        pass

    def setup(self, stage=None):
        
        #merges = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        merges = [[0], [1, 2], [3, 10], [4], [5], [6, 7, 8, 9]]
        self.labels = list(range(len(merges)))
        #self.class_names = ['other',
        #                    'bare ground', 
        #                    'low vegetation',
        #                    'water',
        #                    'building',
        #                    'high vegetation',
        #                    'parking',
        #                    'pedestrian',
        #                    'road',
        #                    'railways',
        #                    'swimmingpool']
        self.class_names = [
            'other',
            'pervious surface',
            'water',
            'building',
            'high vegetation',
            'transport network'
        ]
        self.label_colors = [
            (255, 255, 255),
            (34, 139, 34),
            (0, 0, 238),
            (238, 118, 33),
            (0, 222, 137),
            (38, 38, 38),
        ]
        #self.label_colors = [
        #    (0,0,0),
        #    (100,50,0),
        #    (0,250,50),
        #    (0,50,250),
        #    (250,50,50),
        #    (0,100,50),
        #    (200,200,200),
        #    (200,150,50),
        #    (100,100,100),
        #    (200,100,200),
        #    (50,150,250)
        #]
        
        train_datasets = []
        validation_datasets = []
        with open(self.splitfile_path, newline='') as splitfile:
            self.train_set, self.val_set = build_datasets_from_csv(
                splitfile,
                test_fold=self.test_fold,
                img_aug=self.img_aug,
                data_path=self.data_path,
                merges = merges,
                class_names = self.class_names,
                crop_size = self.crop_size
            )


    def train_dataloader(self):

        train_sampler = RandomSampler(
            data_source=self.train_set,
            replacement=True,
            num_samples=self.epoch_len
        )

        train_dataloader = DataLoader(
            dataset=self.train_set,
            batch_size=self.sup_batch_size,
            collate_fn=CustomCollate(batch_aug=self.batch_aug),
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
            collate_fn=CustomCollate(batch_aug='no'),
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
        unsup_data_path,
        *args,
        **kwargs
    ):
        
        super().__init__(*args, **kwargs)
        self.unsup_batch_size = unsup_batch_size
        self.unsup_crop_size = unsup_crop_size
        self.unsup_data_path = unsup_data_path

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--unsup_batch_size", type=int, default=16)
        parser.add_argument("--unsup_crop_size", type=int, default=160)
        parser.add_argument("--unsup_data_path", type=str, default='')
        return parser

    def setup(self, stage=None):

        super().setup(stage=stage)
        unlabeled_paths = [
            'Toulouse/normalized_mergedTO.tif',
            #'Strasbourg/ORT_P1BPX-2018062038865324CP_epsg32632_decoup.tif',
            #'Biarritz/biarritz_ortho_cropped.tif',
            #'Paris/emprise_ORTHO_cropped.tif',
        ]
        unlabeled_sets = []

        m, M = np.array(DigitanieDs.DATASET_DESC['min'][:3]), np.array(DigitanieDs.DATASET_DESC['max'][:3])
        for path in unlabeled_paths:
            big_raster_path = os.path.join(self.data_path, path)
            width, height = imagesize.get(big_raster_path)
            tile = Window(0, 0, width, height)
            unlabeled_sets.append(
                DigitanieDs(
                    image_path=big_raster_path,
                    tile=tile,
                    fixed_crops=False,
                    read_window_fn=read_window_basic,
                    norm_fn=partial(
                        minmax,
                        m=m,
                        M=M
                    ),
                    crop_size=self.unsup_crop_size,
                    crop_step=self.unsup_crop_size,
                    img_aug=self.img_aug
                )
            )
        
        self.unsup_train_set = ConcatDataset(unlabeled_sets) 

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
            collate_fn=CustomCollate(batch_aug='no'),
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_function
        )

        train_dataloaders = {
            "sup": train_dataloader,
            "unsup": unsup_train_dataloader
        }

        return train_dataloaders 


