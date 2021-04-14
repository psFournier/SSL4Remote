from argparse import ArgumentParser
from functools import partial

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from transforms import MergeLabels

import albumentations as A
from albumentations.pytorch import ToTensorV2
from common_utils.augmentations import get_augmentations
import torch
import numpy as np

class BaseSupervisedDatamodule(LightningDataModule):

    def __init__(self,
                 data_dir,
                 crop_size,
                 nb_pass_per_epoch,
                 batch_size,
                 workers,
                 augmentations,
                 prop_val,
                 prop_train,
                 *args,
                 **kwargs):

        super().__init__()

        self.data_dir = data_dir
        self.crop_size = crop_size
        self.nb_pass_per_epoch = nb_pass_per_epoch
        self.batch_size = batch_size
        self.num_workers = workers
        self.prop_val = prop_val
        self.prop_train = prop_train

        self.train_augment = A.Compose(
            get_augmentations(augmentations) + [
                A.Normalize(),
                ToTensorV2(transpose_mask=False)]
        )
        self.val_augment = ToTensorV2(transpose_mask=False)

        # For binary classification, all labels other than that of interest are collapsed
        self.label_merger = MergeLabels([[0], [1]])

        self.sup_train_set = None
        self.val_set = None


    def prepare_data(self, *args, **kwargs):

        # Nothing to write on disk, data is already there, no hard
        # preprocessing necessary
        pass

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--nb_pass_per_epoch", type=float, default=0.01,
                            help='how many times per epoch the dataset should be spanned')
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--crop_size", type=int, default=128)
        parser.add_argument("--prop_train", type=float, default=0.7)
        parser.add_argument("--prop_val", type=float, default=0.3)
        parser.add_argument("-w", "--workers", default=8, type=int,
                            help="Num workers")
        parser.add_argument('--augmentations', type=str, default='safe')

        return parser

    # Following pytorch Dataloader doc, loading from a map-style dataset is
    # roughly equivalent with:
    #
    #     for indices in batch_sampler:
    #         yield collate_fn([dataset[i] for i in indices])

    def collate_labeled(self, batch, augment):

        # We apply transforms here because transforms are method-dependent
        # while the dataset class should be method independent.
        transformed_batch = [
            augment(
                image=image,
                mask=self.label_merger(ground_truth)
            )
            for image,ground_truth in batch
        ]
        batch = [(elem["image"], elem["mask"]) for elem in transformed_batch]

        return default_collate(batch)

    def wif(self, id):
        uint64_seed = torch.initial_seed()
        np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])

    def train_dataloader(self):

        """
        Contrary to many standard image datasets with a lot of small images,
        remote sensing datasets like ISPRS Vaihingen come with a few big images.
        Thus dataset classes get_item functions provide only a crop of the image.
        For an epoch to actually span around the entirety of the dataset, we thus
        need to sample mutliple times randomly from each big image. Hence the need
        for RandomSamplers and for the nb_pass_per_epoch parameter, otherwise the
        agent would see a single crop from each image during an epoch.
        """

        sup_train_sampler = RandomSampler(
            data_source=self.sup_train_set,
            replacement=True,
            num_samples=int(self.nb_pass_per_epoch * len(self.sup_train_set)),
        )

        # num_workers should be the number of cpus on the machine.
        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.collate_labeled,
                augment=self.train_augment
            ),
            sampler=sup_train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.wif
        )

        return sup_train_dataloader

    def val_dataloader(self):

        num_samples = int(self.nb_pass_per_epoch * len(self.val_set))
        val_sampler = RandomSampler(
            data_source=self.val_set, replacement=True, num_samples=num_samples
        )

        # num_workers should be the number of cpus on the machine.
        val_dataloader = DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            collate_fn=partial(
                self.collate_labeled,
                augment=self.val_augment
            ),
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=self.wif
        )

        return val_dataloader
