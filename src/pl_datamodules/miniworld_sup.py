import numpy as np
from torch.utils.data import ConcatDataset

from pl_datamodules import BaseSupervisedDatamodule


class MiniworldSup(BaseSupervisedDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        sup_train_datasets = []
        val_datasets = []
        city_classes = [
            getattr('datasets', name) for name in [
                'MiniworldParisLabeled',
                'MiniworldArlingtonLabeled'
            ]
        ]

        for city_class in city_classes:

            shuffled_idxs = np.random.permutation(
                len(city_class.labeled_image_paths)
            )

            val_idxs = shuffled_idxs[:self.nb_im_val]
            train_idxs = shuffled_idxs[-self.nb_im_train:]

            sup_train_datasets.append(
                city_class(self.data_dir, train_idxs, self.crop_size)
            )

            val_datasets.append(
                city_class(self.data_dir, val_idxs, self.crop_size)
            )

        self.sup_train_set = ConcatDataset(sup_train_datasets)
        self.val_set = ConcatDataset(val_datasets)