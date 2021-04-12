import random
from torch.utils.data import ConcatDataset
import os

from pl_datamodules import BaseSemisupDatamodule
from torch_datasets import MiniworldCity, MiniworldCityUnlabeled, \
    MiniworldCityLabeled

class MiniworldSemisup(BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        sup_train_datasets = []
        val_datasets = []
        unsup_train_datasets = []
        for city_info in MiniworldCity.city_info_list:

            labeled_image_paths = [
                                      'test/{}_x.png'.format(i) for i in range(city_info[1])
                                  ] + [
                                      'train/{}_x.png'.format(i) for i in range(city_info[2])
                                  ]
            label_paths = [
                              'test/{}_y.png'.format(i) for i in range(city_info[1])
                          ] + [
                              'train/{}_y.png'.format(i) for i in range(city_info[2])
                          ]
            unlabeled_image_paths = []

            nb_labeled_images = len(labeled_image_paths)
            labeled_idxs = list(range(nb_labeled_images))
            # random.shuffle(labeled_idxs)

            sup_train_datasets.append(
                MiniworldCityLabeled(
                    data_path=os.path.join(self.data_dir, city_info[0]),
                    labeled_image_paths=labeled_image_paths,
                    label_paths=label_paths,
                    unlabeled_image_paths=unlabeled_image_paths,
                    image_size=city_info[3],
                    idxs=labeled_idxs[city_info[1]:],
                    crop=self.crop_size
                )
            )
            val_datasets.append(
                MiniworldCityLabeled(
                    data_path=os.path.join(self.data_dir, city_info[0]),
                    labeled_image_paths=labeled_image_paths,
                    label_paths=label_paths,
                    unlabeled_image_paths=unlabeled_image_paths,
                    image_size=city_info[3],
                    idxs=labeled_idxs[:city_info[1]],
                    crop=self.crop_size
                )
            )

            nb_unsup_train_img = len(labeled_image_paths) + len(
                unlabeled_image_paths)
            unsup_train_idxs = list(range(nb_unsup_train_img))
            # random.shuffle(unsup_train_idxs)
            unsup_train_datasets.append(
                MiniworldCityUnlabeled(
                    data_path=os.path.join(self.data_dir, city_info[0]),
                    labeled_image_paths=labeled_image_paths,
                    label_paths=label_paths,
                    unlabeled_image_paths=unlabeled_image_paths,
                    image_size=city_info[3],
                    idxs=unsup_train_idxs,
                    crop=self.crop_size
                )
            )

        self.sup_train_set = ConcatDataset(sup_train_datasets)
        self.val_set = ConcatDataset(val_datasets)
        self.unsup_train_set = ConcatDataset(unsup_train_datasets)