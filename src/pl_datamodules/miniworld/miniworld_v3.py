from pl_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from torch_datasets import MultipleImages, MultipleImagesLabeled
import glob
from torch.utils.data import ConcatDataset
import numpy as np


class MiniworldV3(BaseSupervisedDatamodule):

    """
    MiniworldV3 trainset contains all images from cities in param --train_cities, the val set contains all images from
    cities in param --val_cities.
    """

    def __init__(self, train_cities, val_cities, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.train_cities = train_cities
        self.val_cities = val_cities

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--train_cities", nargs='+', type=str, default=[])
        parser.add_argument("--val_cities", nargs='+', type=str, default=[])

        return parser

    @staticmethod
    def colors_to_labels(colors):

        '''
        Creates one-hot encoded labels for binary classification.
        :param labels_color:
        :return:
        '''

        labels0 = np.zeros(shape=colors.shape[1:], dtype=float)
        labels1 = np.zeros(shape=colors.shape[1:], dtype=float)
        mask = np.any(colors != [0], axis=0)
        np.putmask(labels0, ~mask, 1.)
        np.putmask(labels1, mask, 1.)
        labels = np.stack([labels0, labels1], axis=0)

        return labels

    def setup(self, stage=None):

        city_sup_train_sets = []
        city_val_sets = []

        for city in self.train_cities:

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_y.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_y.tif'))

            sup_train_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list,
                crop=self.crop_size,
                colors_to_labels=self.colors_to_labels
            )
            city_sup_train_sets.append(sup_train_set)

        for city in self.val_cities:

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_y.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_y.tif'))

            sup_val_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list,
                crop=self.crop_size,
                colors_to_labels=self.colors_to_labels
            )
            city_val_sets.append(sup_val_set)

        self.sup_train_set = ConcatDataset(city_sup_train_sets)
        self.val_set = ConcatDataset(city_val_sets)


class MiniworldV3Semisup(MiniworldV3, BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        super(MiniworldV3Semisup, self).setup(stage=stage)

        city_unsup_train_sets = []

        for city in self.train_cities+self.val_cities:

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')) + \
                         sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            unsup_train_set = MultipleImages(
                images_paths=image_list,
                crop=self.crop_size
            )
            city_unsup_train_sets.append(unsup_train_set)

        self.unsup_train_set = ConcatDataset(city_unsup_train_sets)