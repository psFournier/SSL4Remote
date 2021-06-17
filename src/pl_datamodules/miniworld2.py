from pl_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from torch_datasets import MultipleImages, MultipleImagesLabeled
import glob
from torch.utils.data import ConcatDataset


class Miniworld2(BaseSupervisedDatamodule):

    def __init__(self, cities, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cities = cities

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--cities", nargs='+', type=str, default=[])

        return parser

    def setup(self, stage=None):

        city_sup_train_sets = []
        city_val_sets = []

        for city in self.cities:

            image_list = glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')
            label_list = glob.glob(f'{self.data_dir}/{city}/train/*_y.tif')
            sup_train_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list
            )
            city_sup_train_sets.append(sup_train_set)

            image_list = glob.glob(f'{self.data_dir}/{city}/test/*_x.tif')
            label_list = glob.glob(f'{self.data_dir}/{city}/test/*_y.tif')
            val_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list
            )
            city_val_sets.append(val_set)

        self.sup_train_set = ConcatDataset(city_sup_train_sets)
        self.val_set = ConcatDataset(city_val_sets)


class Miniworld2Semisup(BaseSemisupDatamodule):

    def __init__(self, cities, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cities = cities

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--cities", nargs='+', type=str, default=[])

        return parser

    def setup(self, stage=None):

        city_sup_train_sets = []
        city_val_sets = []
        city_unsup_train_sets = []

        for city in self.cities:

            image_list = glob.glob(f'{self.data_dir}/{city}/train/*_x.tif')
            label_list = glob.glob(f'{self.data_dir}/{city}/train/*_y.tif')
            sup_train_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list
            )
            city_sup_train_sets.append(sup_train_set)

            image_list = glob.glob(f'{self.data_dir}/{city}/test/*_x.tif')
            label_list = glob.glob(f'{self.data_dir}/{city}/test/*_y.tif')
            val_set = MultipleImagesLabeled(
                images_paths=image_list,
                labels_paths=label_list
            )
            city_val_sets.append(val_set)

            image_list = glob.glob(f'{self.data_dir}/{city}/train/*_x.tif') + \
                         glob.glob(f'{self.data_dir}/{city}/test/*_x.tif')
            unsup_train_set = MultipleImages(
                images_paths=image_list
            )
            city_unsup_train_sets.append(unsup_train_set)

        self.sup_train_set = ConcatDataset(city_sup_train_sets)
        self.val_set = ConcatDataset(city_val_sets)
        self.unsup_train_set = ConcatDataset(city_unsup_train_sets)