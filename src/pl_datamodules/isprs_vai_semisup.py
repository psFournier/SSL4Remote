from argparse import ArgumentParser

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data._utils.collate import default_collate

from datasets import IsprsVaihingen, IsprsVaihingenLabeled, IsprsVaihingenUnlabeled
from transforms import MergeLabels
from pl_datamodules import IsprsVaiSup

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pl_datamodules import BaseSemisupDatamodule

class IsprsVaiSemisup(BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        shuffled_idxs = list(
            np.random.permutation(
                len(IsprsVaihingen.labeled_image_paths)
            )
        )

        val_idxs = shuffled_idxs[:self.nb_im_val]
        train_idxs = shuffled_idxs[-self.nb_im_train:]

        self.sup_train_set = IsprsVaihingenLabeled(
            self.data_dir, train_idxs, self.crop_size
        )

        self.val_set = IsprsVaihingenLabeled(
            self.data_dir, val_idxs, self.crop_size
        )

        # ...but each non validation labeled image can be used without its
        # label for unsupervised training
        unlabeled_idxs = list(range(len(IsprsVaihingen.unlabeled_image_paths)))
        all_unsup_train_idxs = shuffled_idxs[self.nb_im_val:] + \
                              unlabeled_idxs
        unsup_train_idxs = all_unsup_train_idxs[:self.nb_im_unsup_train]
        self.unsup_train_set = IsprsVaihingenUnlabeled(
            self.data_dir,
            unsup_train_idxs,
            self.crop_size
        )