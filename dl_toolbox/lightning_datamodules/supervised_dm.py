from argparse import ArgumentParser
import os
import csv

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window

from dl_toolbox.utils import worker_init_function, build_split_from_csv, read_splitfile
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_datasets import *


dataset_cls_dict = {
    'DigitanieToulouseDs': DigitanieToulouseDs
}

class SupervisedDm(LightningDataModule):

    def __init__(self,
                 data_path,
                 dataset_cls,
                 labels,
                 label_merger,
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
        self.dataset_cls = dataset_cls_dict[dataset_cls]
        self.labels = labels
        self.label_merger = label_merger
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
        parser.add_argument("--labels", type=str)
        parser.add_argument("--label_merger", type=str)
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
        
        #with open(self.splitfile_path, newline='') as splitfile:
        #    train_sets, val_sets = build_split_from_csv(
        #        splitfile=splitfile,
        #        dataset_cls=self.dataset_cls,
        #        train_folds=self.train_folds,
        #        test_folds=self.test_folds,
        #        img_aug=self.img_aug,
        #        data_path=self.data_path,
        #        crop_size = self.crop_size,
        #        one_hot=True
        #    )
        #if train_sets: self.train_set = ConcatDataset(train_sets)
        #if val_sets: self.val_set = ConcatDataset(val_sets)

        with open(self.splitfile_path, newline='') as splitfile:
            train_args, val_args = read_splitfile(
                splitfile=splitfile,
                data_path=self.data_path,
                train_folds=self.train_folds,
                test_folds=self.test_folds
            )

        if train_args:
            self.train_set = ConcatDataset([
                self.dataset_cls(
                    labels=self.labels,
                    label_merger=self.label_merger,
                    img_aug=self.img_aug,
                    crop_size=self.crop_size,
                    crop_step=self.crop_size,
                    one_hot=True,
                    **kwarg
                ) for kwarg in train_args
            ])

        if val_args:
            self.val_set = ConcatDataset([
                self.dataset_cls(
                    labels=self.labels,
                    label_merger=self.label_merger,
                    img_aug='no',
                    crop_size=self.crop_size,
                    crop_step=self.crop_size,
                    one_hot=True,
                    **kwarg
                ) for kwarg in val_args
            ])

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

