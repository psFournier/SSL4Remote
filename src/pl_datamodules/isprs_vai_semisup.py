import numpy as np
import random

from torch_datasets import IsprsV, IsprsVLabeled, IsprsVUnlabeled
from transforms import MergeLabels

from pl_datamodules import BaseSemisupDatamodule
import albumentations as A

class IsprsVaiSemisup(BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels([[0], [1]])
        self.unsup_train_augment = A.NoOp()

    def setup(self, stage=None):

        nb_labeled_images = len(IsprsV.labeled_image_paths)
        labeled_idxs = list(range(nb_labeled_images))
        # random.shuffle(labeled_idxs)

        nb_val_img = int(nb_labeled_images * self.prop_val)
        nb_train_img = int(nb_labeled_images * self.prop_train)
        val_idxs = labeled_idxs[:nb_val_img]
        train_idxs = labeled_idxs[-nb_train_img:]

        self.sup_train_set = IsprsVLabeled(
            data_path=self.data_dir,
            idxs=train_idxs,
            crop=self.crop_size,
            label_merger=self.label_merger,
            augmentations=self.train_augment
        )

        self.val_set = IsprsVLabeled(
            data_path=self.data_dir,
            idxs=val_idxs,
            crop=self.crop_size,
            label_merger=self.label_merger,
            augmentations=self.val_augment
        )

        # ...but each non validation labeled image can be used without its
        # label for unsupervised training
        nb_unsup_train_img = len(IsprsV.labeled_image_paths) + len(
            IsprsV.unlabeled_image_paths)
        unsup_train_idxs = list(range(nb_unsup_train_img))

        self.unsup_train_set = IsprsVUnlabeled(
            data_path=self.data_dir,
            idxs=unsup_train_idxs,
            crop=self.crop_size,
            augmentations=self.unsup_train_augment
        )