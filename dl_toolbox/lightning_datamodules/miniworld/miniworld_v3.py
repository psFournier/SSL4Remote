from dl_toolbox.lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from dl_toolbox.torch_datasets import MiniworldCityDs
import glob
from torch.utils.data import ConcatDataset
import numpy as np
from functools import partial


class MiniworldDmV3(BaseSupervisedDatamodule):

    """
    MiniworldV3 trainset contains all images from cities in param --train_cities, the val set contains all images from
    cities in param --val_cities.
    """

    def __init__(self, data_dir, train_cities, val_cities, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.train_cities = train_cities
        self.val_cities = val_cities

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--train_cities", nargs='+', type=str, default=[])
        parser.add_argument("--val_cities", nargs='+', type=str, default=[])

        return parser

    def setup(self, stage=None):

        city_sup_train_sets = []
        city_val_sets = []

        for city in self.train_cities:

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_y.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_y.tif'))

            sup_train_set = MiniworldCityDs(
                city=city,
                images_paths=image_list,
                labels_paths=label_list,
                crop_size=self.crop_size,
                transforms=self.train_dataset_transforms,
            )
            city_sup_train_sets.append(sup_train_set)

        for city in self.val_cities:

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_y.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_y.tif'))

            sup_val_set = MiniworldCityDs(
                city=city,
                images_paths=image_list,
                labels_paths=label_list,
                crop_size=self.crop_size,
                transforms=self.train_dataset_transforms,
            )
            city_val_sets.append(sup_val_set)

        self.sup_train_set = ConcatDataset(city_sup_train_sets)
        self.val_set = ConcatDataset(city_val_sets)


class MiniworldDmV3Semisup(MiniworldDmV3, BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        super(MiniworldDmV3Semisup, self).setup(stage=stage)

        city_unsup_train_sets = []

        for city in self.train_cities+self.val_cities:

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            unsup_train_set = MiniworldCityDs(
                city=city,
                images_paths=image_list,
                crop_size=self.crop_size,
                transforms=self.train_dataset_transforms
            )
            city_unsup_train_sets.append(unsup_train_set)

        self.unsup_train_set = ConcatDataset(city_unsup_train_sets)