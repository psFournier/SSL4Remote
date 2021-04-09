import numpy as np
import random

from torch_datasets import IsprsVaihingen, IsprsVaihingenLabeled, IsprsVaihingenUnlabeled

from pl_datamodules import BaseSemisupDatamodule


class IsprsVaiSemisup(BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        nb_labeled_images = len(IsprsVaihingen.labeled_image_paths)
        labeled_idxs = list(range(nb_labeled_images))
        random.shuffle(labeled_idxs)

        val_idxs = labeled_idxs[:self.nb_im_val]
        train_idxs = labeled_idxs[-self.nb_im_train:]

        self.sup_train_set = IsprsVaihingenLabeled(
            self.data_dir, train_idxs, self.crop_size
        )

        self.val_set = IsprsVaihingenLabeled(
            self.data_dir, val_idxs, self.crop_size
        )

        # ...but each non validation labeled image can be used without its
        # label for unsupervised training
        nb_unlabeled_images = len(IsprsVaihingen.unlabeled_image_paths)
        unlabeled_idxs = list(range(nb_unlabeled_images))
        unlabeled_idxs = [nb_labeled_images+i for i in unlabeled_idxs]

        all_unsup_train_idxs = labeled_idxs[self.nb_im_val:] + unlabeled_idxs
        random.shuffle(all_unsup_train_idxs)
        unsup_train_idxs = all_unsup_train_idxs[:self.nb_im_unsup_train]
        self.unsup_train_set = IsprsVaihingenUnlabeled(
            self.data_dir,
            unsup_train_idxs,
            self.crop_size
        )