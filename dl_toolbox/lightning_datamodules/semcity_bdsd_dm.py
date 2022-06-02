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

from dl_toolbox.utils import worker_init_function, get_tiles
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import SemcityBdsdDs
from dl_toolbox.torch_datasets.utils import *

class SemcityBdsdDm(LightningDataModule):

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
        parser.add_argument("--test_fold", type=int)
        parser.add_argument("--epoch_len", type=int, default=10000)
        parser.add_argument("--sup_batch_size", type=int, default=16)
        parser.add_argument("--crop_size", type=int, default=128)
        parser.add_argument("--workers", default=6, type=int)
        parser.add_argument('--img_aug', type=str, default='no')
        parser.add_argument('--batch_aug', type=str, default='no')

        return parser

    def prepare_data(self, *args, **kwargs):

        pass

    def setup(self, stage=None):

        self.label_colors = SemcityBdsdDs.DATASET_DESC['label_colors']
        self.labels = list(range(len(self.label_colors)))
        self.class_names = [info[1] for info in SemcityBdsdDs.DATASET_DESC['labels']]
         
        train_datasets = []
        validation_datasets = []
        with open(self.splitfile_path, newline='') as splitfile:
            reader = csv.reader(splitfile)
            next(reader) # skipping header
            for row in reader:
                is_val = int(row[8])==self.test_fold
                aug = 'no' if is_val else self.img_aug
                window = Window(
                    col_off=int(row[4]),
                    row_off=int(row[5]),
                    width=int(row[6]),
                    height=int(row[7])
                )
                dataset = SemcityBdsdDs(
                    image_path=os.path.join(self.data_path, row[2]),
                    label_path=os.path.join(self.data_path, row[3]),
                    fixed_crops=is_val,
                    tile=window,
                    crop_size=self.crop_size,
                    crop_step=self.crop_size,
                    read_window_fn=partial(
                        read_window_from_big_raster_gdal, 
                        raster_path=os.path.join(data_path, row[9])
                    ),
                    norm_fn=partial(
                        minmax,
                        m=m,
                        M=M
                    ),
                    img_aug=aug
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

class SemcityBdsdDmSemisup(SemcityBdsdDm):

    def __init__(self, unsup_batch_size, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.unsup_batch_size = unsup_batch_size
        self.unsup_crop_size = unsup_crop_size

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--unsup_batch_size', type=int)
        parser.add_argument('--unsup_crop_size', type=int)

        return parser

    def setup(self, stage=None):

        super(SemcityBdsdDmSemisup, self).setup(stage=stage)
        nums = ['09','12','06','01','05','11','10','02','14','15','13','16']
        image_paths = [f'{self.data_path}/test/TLS_BDSD_M_{num}.tif' for num in nums]
        unsup_train_sets = [
            SemcityBdsdDs(
                image_path=image_path,
                tile=Window(0, 0, *imagesize.get(image_path)),
                fixed_crops=False,
                crop_size=self.unsup_crop_size,
                crop_step=self.unsup_crop_size,
                img_aug=self.img_aug
            ) for image_path in image_paths
        ]
        self.unsup_train_set = ConcatDataset(unsup_train_sets)

def main():

    datamodule = SemcityBdsdDm(
        data_path='/d/pfournie/ai4geo/data/SemcityTLS_DL',
        splitfile_path='/d/pfournie/ai4geo/split_semcity.csv',
        test_fold=4,
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

if __name__ == '__main__':

    main()



