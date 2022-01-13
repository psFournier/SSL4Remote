from lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from torch_datasets import MiniworldCityDs
import glob
from torch.utils.data import ConcatDataset
import numpy as np
from functools import partial


class MiniworldDmV2(BaseSupervisedDatamodule):

    """
    The train set is the merge of the train folders of all cities in param --cities, the val set is the merge of the
    test folders of all cities in param --cities.
    """

    def __init__(self, data_dir, cities, label_decrease_factor, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.cities = cities
        self.label_decrease_factor = label_decrease_factor

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--cities", nargs='+', type=str, default=[])
        parser.add_argument("--label_decrease_factor", type=int, default=1)

        return parser

    def setup(self, stage=None):

        city_sup_train_sets = []
        city_val_sets = []

        for city in self.cities:

            # Attention on s'appuie sur sorted pour s'assurer de la correspondance image-label: pas très robuste.
            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/train/*_y.tif'))
            sup_train_set = MiniworldCityDs(
                city=city,
                images_paths=image_list[::self.label_decrease_factor],
                labels_paths=label_list[::self.label_decrease_factor],
                crop_size=self.crop_size,
                img_aug=self.img_aug,
            )
            city_sup_train_sets.append(sup_train_set)

            image_list = sorted(glob.glob(f'{self.data_dir}/{city}/test/*_x.tif'))
            label_list = sorted(glob.glob(f'{self.data_dir}/{city}/test/*_y.tif'))
            val_set = MiniworldCityDs(
                city=city,
                images_paths=image_list,
                labels_paths=label_list,
                crop_size=self.crop_size,
                img_aug=self.img_aug,
            )
            city_val_sets.append(val_set)

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

class MiniworldDmV2Semisup(MiniworldDmV2, BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        super(MiniworldDmV2Semisup, self).setup(stage=stage)

        city_unsup_train_sets = []

        for city in self.cities:

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
