from pytorch_lightning import LightningDataModule
import numpy as np
from datasets import Isprs_labeled, Isprs_unlabeled
from torch.utils.data import DataLoader, RandomSampler
from argparse import ArgumentParser

# from samplers import Multiple_pass

class Isprs_semisup(LightningDataModule):

    def __init__(self,
                 data_path,
                 crop_size,
                 nb_pass_per_epoch,
                 batch_size,
                 sup_train_transforms,
                 val_transforms,
                 unsup_train_transforms
                 ):

        super().__init__()
        self.labeled_idxs = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
                          34, 37]
        self.unlabeled_idxs = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29,
                               31, 33, 35, 38]
        self.data_path = data_path
        self.crop_size = crop_size
        self.nb_pass_per_epoch = nb_pass_per_epoch
        self.batch_size = batch_size
        self.sup_train_transforms = sup_train_transforms
        self.val_transforms = val_transforms
        self.unsup_train_transforms = unsup_train_transforms

    def prepare_data(self, *args, **kwargs):

        # Nothing to write on disk, data is already there, no hard
        # preprocessing necessary
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nb_pass_per_epoch',
                            type=int,
                            default=1)
        parser.add_argument("--data_dir",
                            type=str,
                            default='/home/pierre/Documents/ONERA/ai4geo/')
        parser.add_argument('--batch_size',
                            type=int,
                            default=16)
        parser.add_argument('--crop_size',
                            type=int,
                            default=128)

        return parser

    def setup(self, stage=None):

        np.random.shuffle(self.labeled_idxs)
        val_idxs = self.labeled_idxs[:7]
        train_idxs = self.labeled_idxs[14:]

        self.sup_train_set = Isprs_labeled(self.data_path,
                                           train_idxs,
                                           self.crop_size,
                                           self.sup_train_transforms)

        self.val_set = Isprs_labeled(self.data_path,
                                     val_idxs,
                                     self.crop_size,
                                     self.val_transforms)

        unsup_train_idxs = self.labeled_idxs[7:] + self.unlabeled_idxs
        self.unsup_train_set = Isprs_unlabeled(self.data_path,
                                               unsup_train_idxs,
                                               self.crop_size,
                                               self.unsup_train_transforms)

    def train_dataloader(self):

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=self.nb_pass_per_epoch*len(self.sup_train_set)
        )
        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            sampler=sup_train_sampler,
            num_workers=2,
            pin_memory=True
        )

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.nb_pass_per_epoch*len(self.unsup_train_set)
        )
        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.batch_size,
            sampler=unsup_train_sampler,
            num_workers=2,
            pin_memory=True
        )

        train_dataloaders = {
            'sup': sup_train_dataloader,
            'unsup': unsup_train_dataloader
        }

        return train_dataloaders

    def val_dataloader(self):

        num_samples = self.nb_pass_per_epoch * len(self.val_set)
        val_sampler = RandomSampler(
            data_source=self.val_set,
            replacement=True,
            num_samples=num_samples
        )
        val_dataloader = DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )

        return val_dataloader