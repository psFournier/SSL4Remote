import random
from torch.utils.data import ConcatDataset
import os

from pl_datamodules import BaseSemisupDatamodule
from torch_datasets import (
    MiniworldParisLabeled, MiniworldParisUnlabeled,
    MiniworldArlingtonLabeled, MiniworldArlingtonUnlabeled
)

class MiniworldSemisup(BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        city_classes = [
            (MiniworldParisLabeled, MiniworldParisUnlabeled),
            (MiniworldArlingtonLabeled, MiniworldArlingtonUnlabeled)
        ]
        dirnames = [
            'paris',
            'Arlington'
        ]

        sup_train_datasets = []
        val_datasets = []
        unsup_train_datasets = []
        for city_class, dirname in zip(city_classes, dirnames):

            nb_labeled_images = len(city_class[0].labeled_image_paths)
            labeled_idxs = list(range(nb_labeled_images))
            random.shuffle(labeled_idxs)

            val_idxs = labeled_idxs[:self.nb_im_val]
            train_idxs = labeled_idxs[-self.nb_im_train:]

            sup_train_datasets.append(
                city_class[0](os.path.join(self.data_dir, dirname), train_idxs,
                           self.crop_size)
            )

            val_datasets.append(
                city_class[0](os.path.join(self.data_dir, dirname), val_idxs, self.crop_size)
            )

            nb_unlabeled_images = len(city_class[1].unlabeled_image_paths)
            unlabeled_idxs = list(range(nb_unlabeled_images))
            unlabeled_idxs = [nb_labeled_images+i for i in unlabeled_idxs]

            all_unsup_train_idxs = labeled_idxs[self.nb_im_val:] + unlabeled_idxs
            random.shuffle(all_unsup_train_idxs)
            unsup_train_idxs = all_unsup_train_idxs[:self.nb_im_unsup_train]
            unsup_train_datasets.append(
                city_class[1](os.path.join(self.data_dir, dirname),unsup_train_idxs,self.crop_size)
            )

        self.sup_train_set = ConcatDataset(sup_train_datasets)
        self.val_set = ConcatDataset(val_datasets)
        self.unsup_train_set = ConcatDataset(unsup_train_datasets)