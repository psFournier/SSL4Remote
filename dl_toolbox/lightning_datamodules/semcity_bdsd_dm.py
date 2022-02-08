from argparse import ArgumentParser
import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np
import imagesize

from dl_toolbox.utils import worker_init_function, get_tiles
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import SemcityBdsdDs


class SemcityBdsdDm(LightningDataModule):

    def __init__(self,
                 image_path,
                 label_path,
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
        self.image_path = image_path
        self.class_names = [label[2] for label in SemcityBdsdDs.labels_desc]
        self.label_path = label_path
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
        parser.add_argument("--image_path", type=str)
        parser.add_argument("--label_path", type=str)
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
        
        w, h = imagesize.get(self.image_path)
        train_val_tiles = list(get_tiles(w, h, size=876, size2=863))
        train_tiles = train_val_tiles[::3] + train_val_tiles[1::3]
        val_tiles = train_val_tiles[2::3]
        
        train_tiles_datasets = [SemcityBdsdDs(
                image_path=self.image_path,
                label_path=self.label_path,
                tile=tile,
                fixed_crops=False,
                crop_size=self.crop_size,
                img_aug=self.img_aug
            ) for tile in train_tiles]
        self.train_set = ConcatDataset(train_tiles_datasets)

        val_tiles_datasets = [SemcityBdsdDs(
                image_path=self.image_path,
                label_path=self.label_path,
                tile=tile,
                fixed_crops=True,
                crop_size=self.crop_size,
                img_aug=self.img_aug
            ) for tile in val_tiles]
        self.val_set = ConcatDataset(val_tiles_datasets)

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
        for val, color, _, _ in SemcityBdsdDs.labels_desc:
            mask = np.array(labels == val)
            rgb_label[mask] = np.array(color)
        rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

        return rgb_label

def main():

    datamodule = SemcityBdsdDm(
        image_path='/home/pfournie/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif',
        label_path='/home/pfournie/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif',
        crop_size=128,
        epoch_len=100,
        sup_batch_size=16,
        workers=0,
        img_aug='no',
        batch_aug='no',
        ignore_void=True
    )

    for batch in datamodule.train_dataloader:

        print(batch['image'].shape)

if __name__ == '__main__':

    main()


#class SemcityBdsdDmSemisup(SemcityBdsdDm, BaseSemisupDatamodule):
#
#    def __init__(self, data_dir, *args, **kwargs):
#
#        super().__init__(*args, **kwargs)
#        self.data_dir = data_dir
#
#    @classmethod
#    def add_model_specific_args(cls, parent_parser):
#
#        parser = super().add_model_specific_args(parent_parser)
#        parser.add_argument('--data_dir', type=str)
#
#        return parser
#
#    def setup(self, stage=None):
#
#        super(SemcityBdsdDmSemisup, self).setup(stage=stage)
#        nums = ['09','12','06','01','05','11','10','02','14','15','13','16']
#        image_paths = [f'{self.data_dir}/TLS_BDSD_M_{num}.tif' for num in nums]
#        unsup_train_sets = []
#        for image_path in image_paths:
#            set = SemcityBdsdDs(
#                image_path=image_path,
#                tile_size=(863, 876),
#                crop_size=self.unsup_crop_size
#            )
#            unsup_train_sets.append(set)
#        self.unsup_train_set = ConcatDataset(unsup_train_sets)
#
