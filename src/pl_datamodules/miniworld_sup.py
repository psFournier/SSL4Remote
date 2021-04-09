import random
from torch.utils.data import ConcatDataset
import os

from pl_datamodules import BaseSupervisedDatamodule
from torch_datasets import (
    MiniworldParisLabeled,
    MiniworldArlingtonLabeled
)


class MiniworldSup(BaseSupervisedDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        city_classes = [MiniworldParisLabeled, MiniworldArlingtonLabeled]
        dirnames = [
            'paris',
            'Arlington'
        ]

        sup_train_datasets = []
        val_datasets = []
        for city_class, dirname in zip(city_classes, dirnames):

            nb_labeled_images = len(city_class.labeled_image_paths)
            labeled_idxs = list(range(nb_labeled_images))
            random.shuffle(labeled_idxs)

            val_idxs = labeled_idxs[:self.nb_im_val]
            train_idxs = labeled_idxs[-self.nb_im_train:]

            sup_train_datasets.append(
                city_class(os.path.join(self.data_dir, dirname), train_idxs,
                           self.crop_size)
            )

            val_datasets.append(
                city_class(self.data_dir, val_idxs, self.crop_size)
            )

        self.sup_train_set = ConcatDataset(sup_train_datasets)
        self.val_set = ConcatDataset(val_datasets)