from argparse import ArgumentParser
import os
from functools import partial
import csv

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
import torch
import numpy as np
import imagesize
from rasterio.windows import Window

from dl_toolbox.utils import worker_init_function, build_split_from_csv
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *
from dl_toolbox.torch_datasets.utils import *


class SupervisedDm(LightningDataModule):

    def __init__(self,
                 data_path,
                 dataset_cls,
                 splitfile_path,
                 test_folds,
                 train_folds,
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
        self.dataset_cls = dataset_cls
        self.splitfile_path = splitfile_path
        self.test_folds = test_folds
        self.train_folds = train_folds
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
        parser.add_argument("--dataset_cls", type=str)
        parser.add_argument("--splitfile_path", type=str)
        parser.add_argument("--test_folds", nargs='+', type=int)
        parser.add_argument("--train_folds", nargs='+', type=int)
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
        
        train_datasets = []
        validation_datasets = []
        with open(self.splitfile_path, newline='') as splitfile:
            self.train_set, self.val_set = build_split_from_csv(
                splitfile=splitfile,
                dataset_cls=self.dataset_cls,
                train_folds=self.train_folds,
                test_folds=self.test_folds,
                img_aug=self.img_aug,
                data_path=self.data_path,
                crop_size = self.crop_size,
                one_hot=True
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
    

#class DigitanieSemisupDm(DigitanieDm):
#
#    def __init__(
#        self,
#        unsup_batch_size,
#        unsup_crop_size,
#        unsup_data_path,
#        *args,
#        **kwargs
#    ):
#        
#        super().__init__(*args, **kwargs)
#        self.unsup_batch_size = unsup_batch_size
#        self.unsup_crop_size = unsup_crop_size
#        self.unsup_data_path = unsup_data_path
#
#    @classmethod
#    def add_model_specific_args(cls, parent_parser):
#
#        parser = super().add_model_specific_args(parent_parser)
#        parser.add_argument("--unsup_batch_size", type=int, default=16)
#        parser.add_argument("--unsup_crop_size", type=int, default=160)
#        parser.add_argument("--unsup_data_path", type=str, default='')
#        return parser
#
#    def setup(self, stage=None):
#
#        super().setup(stage=stage)
#        unlabeled_paths = [
#            #('Toulouse','normalized_mergedTO.tif'),
#            #('Strasbourg','ORT_P1BPX-2018062038865324CP_epsg32632_decoup.tif'),
#            #('Biarritz','biarritz_ortho_cropped.tif'),
#            #('Paris','emprise_ORTHO_cropped.tif'),
#            #('Montpellier','montpellier_ign_cropped.tif')
#            ('Toulouse','toulouse_full_tiled.tif'),
#            ('Strasbourg','strasbourg_full_tiled.tif'),
#            ('Biarritz','biarritz_full_tiled.tif'),
#            ('Paris','paris_full_tiled.tif'),
#            ('Montpellier','montpellier_full_tiled.tif')
#
#        ]
#        unlabeled_sets = []
#
#        for path in unlabeled_paths:
#            m = DigitanieDs.DATASET_DESC['min'][path[0]][:3]
#            M = DigitanieDs.DATASET_DESC['max'][path[0]][:3]
#            big_raster_path = os.path.join(self.data_path, path[1])
#            width, height = imagesize.get(big_raster_path)
#            tile = Window(0, 0, width, height)
#            unlabeled_sets.append(
#                DigitanieDs(
#                    image_path=big_raster_path,
#                    tile=tile,
#                    fixed_crops=False,
#                    read_window_fn=read_window_basic_gdal,
#                    norm_fn=partial(
#                        minmax,
#                        m=m,
#                        M=M
#                    ),
#                    crop_size=self.unsup_crop_size,
#                    crop_step=self.unsup_crop_size,
#                    img_aug=self.img_aug
#                )
#            )
#        
#        self.unsup_train_set = ConcatDataset(unlabeled_sets) 
#
#    def train_dataloader(self):
#
#        train_dataloader = super().train_dataloader()
#        unsup_train_sampler = RandomSampler(
#            data_source=self.unsup_train_set,
#            replacement=True,
#            num_samples=self.epoch_len
#        )
#
#        unsup_train_dataloader = DataLoader(
#            dataset=self.unsup_train_set,
#            batch_size=self.unsup_batch_size,
#            sampler=unsup_train_sampler,
#            collate_fn=CustomCollate(batch_aug='no'),
#            num_workers=self.num_workers,
#            pin_memory=True,
#            worker_init_fn=worker_init_function
#        )
#
#        train_dataloaders = {
#            "sup": train_dataloader,
#            "unsup": unsup_train_dataloader
#        }
#
#        return train_dataloaders

def main():

    datamodule = SupervisedDm(
        dataset_cls=SemcityBdsdDs,
        data_path='/d/pfournie/ai4geo/data/SemcityTLS_DL',
        splitfile_path='/d/pfournie/ai4geo/split_semcity.csv',
        #data_path='/d/pfournie/ai4geo/data/DIGITANIE',
        #splitfile_path='/d/pfournie/ai4geo/split_toulouse.csv',
        test_folds=(4,),
        train_folds=(0,1,2,3),
        crop_size=128,
        epoch_len=100,
        sup_batch_size=16,
        workers=0,
        img_aug='d4_color-0',
        batch_aug='no',
    )

    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    for batch in dataloader:

        print(batch['image'].shape)
        print(batch['mask'].shape)

if __name__ == '__main__':

    main()

