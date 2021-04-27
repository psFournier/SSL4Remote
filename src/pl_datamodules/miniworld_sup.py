from pl_datamodules import BaseSupervisedDatamodule
from torch_datasets.miniworld import *
from torch import tensor

cities = {
    'christchurch': (Christchurch, ChristchurchLabeled),
    'paris': (Paris, ParisLabeled)
}

class MiniworldSup(BaseSupervisedDatamodule):

    class_weights = tensor(
        [1., 1.]
    )

    def __init__(self, city, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.city = city

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--city", type=str, default='christchurch')

        return parser

    def setup(self, stage=None):

        nb_labeled_images = cities[self.city][0].nb_labeled_images
        labeled_idxs = list(range(nb_labeled_images))

        train, val = cities[self.city][0].default_train_val
        self.sup_train_set = cities[self.city][1](
            data_path=self.data_dir,
            idxs=labeled_idxs[:train][::self.prop_train],
            crop=self.crop_size,
            augmentations=self.train_augment
        )

        self.val_set = cities[self.city][1](
            data_path=self.data_dir,
            idxs=labeled_idxs[train:],
            crop=self.crop_size,
            augmentations=self.val_augment
        )