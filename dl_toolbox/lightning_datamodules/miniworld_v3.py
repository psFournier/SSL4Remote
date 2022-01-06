from lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from torch_datasets import MiniworldCityDs
import glob
from torch.utils.data import ConcatDataset
import numpy as np
from functools import partial
import torch


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
                img_aug=self.img_aug,
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
                img_aug=self.img_aug,
            )
            city_val_sets.append(sup_val_set)

        self.sup_train_set = ConcatDataset(city_sup_train_sets)
        self.val_set = ConcatDataset(city_val_sets)

    @property
    def class_names(self):
        return ['non building', 'building']

    def label_to_rgb(self, labels):

        rgb_label = np.zeros(shape=(*labels.shape, 3), dtype=float)
        mask = np.array(labels == 1)
        rgb_label[mask] = np.array([255,255,255])
        rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

        return rgb_label


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
                crop_size=self.unsup_crop_size,
                img_aug=self.img_aug
            )
            city_unsup_train_sets.append(unsup_train_set)

        self.unsup_train_set = ConcatDataset(city_unsup_train_sets)
