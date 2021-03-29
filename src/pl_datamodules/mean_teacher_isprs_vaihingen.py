from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from datasets import IsprsVaihingen, IsprsVaihingenLabeled, IsprsVaihingenUnlabeled


class MeanTeacherIsprsVaihingen(LightningDataModule):
    def __init__(
        self,
        data_path,
        crop_size,
        nb_pass_per_epoch,
        batch_size,
        sup_train_transforms,
        val_transforms,
        unsup_train_transforms,
    ):

        super().__init__()

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
        parser.add_argument("--nb_pass_per_epoch", type=int, default=1)
        parser.add_argument(
            "--data_dir", type=str, default="/home/pierre/Documents/ONERA/ai4geo/"
        )
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--crop_size", type=int, default=128)

        return parser

    def setup(self, stage=None):

        labeled_idxs = IsprsVaihingen.labeled_idxs
        np.random.shuffle(labeled_idxs)

        # Here we use very few labeled images for training (2)...
        val_idxs = labeled_idxs[:7]
        train_idxs = labeled_idxs[15:]

        self.sup_train_set = IsprsVaihingenLabeled(
            self.data_path, train_idxs, self.crop_size,
            transforms=None
        )

        self.val_set = IsprsVaihingenLabeled(
            self.data_path, val_idxs, self.crop_size,
            transforms=None
        )

        # ...but each non validation labeled image is used without its label for
        # unsupervised training
        unlabeled_idxs = IsprsVaihingen.unlabeled_idxs
        unsup_train_idxs = labeled_idxs[7:] + unlabeled_idxs
        self.unsup_train_set = IsprsVaihingenUnlabeled(
            self.data_path,
            unsup_train_idxs,
            self.crop_size,
            transforms=None,
        )

    # Following pytorch Dataloader doc, loading from a map-style dataset is
    # roughly equivalent with:
    #
    #     for indices in batch_sampler:
    #         yield collate_fn([dataset[i] for i in indices])

    def collate_labeled(self, batch):

        transformed_batch = [self.sup_train_transforms(image=image,
                                                       mask=ground_truth) for
                             image,ground_truth in batch]
        batch = [(elem["image"], elem["mask"]) for elem in transformed_batch]

        return default_collate(batch)

    def collate_unlabeled(self, batch):

        transformed_batch = [self.unsup_train_transforms(image=image) for
                             image in batch]
        batch = [(elem["image"]) for elem in transformed_batch]

        return default_collate(batch)

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
            num_samples=self.nb_pass_per_epoch * len(self.sup_train_set),
        )

        # num_workers should be the number of cpus on the machine.
        sup_train_dataloader = DataLoader(
            dataset=self.sup_train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_labeled,
            sampler=sup_train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        unsup_train_sampler = RandomSampler(
            data_source=self.unsup_train_set,
            replacement=True,
            num_samples=self.nb_pass_per_epoch * len(self.unsup_train_set),
        )
        print(self.nb_pass_per_epoch * len(self.unsup_train_set))
        # num_workers should be the number of cpus on the machine.
        unsup_train_dataloader = DataLoader(
            dataset=self.unsup_train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_unlabeled,
            sampler=unsup_train_sampler,
            num_workers=4,
            pin_memory=True,
        )

        train_dataloaders = {
            "sup": sup_train_dataloader,
            "unsup": unsup_train_dataloader,
        }

        return train_dataloaders

    def val_dataloader(self):

        num_samples = self.nb_pass_per_epoch * len(self.val_set)
        val_sampler = RandomSampler(
            data_source=self.val_set, replacement=True, num_samples=num_samples
        )

        # num_workers should be the number of cpus on the machine.
        val_dataloader = DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_labeled,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
        )

        return val_dataloader
