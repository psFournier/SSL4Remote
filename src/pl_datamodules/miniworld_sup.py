import random
from torch.utils.data import ConcatDataset
import os

from pl_datamodules import BaseSupervisedDatamodule
from torch_datasets import MiniworldCity

city_info_list = [
    ('Arlington', 15, 20, (3000,3000)),
    ('Atlanta', 1, 2, (1800,1200)),
    ('austin', 15, 20, (3000,3000)),
    ('Austin', 1, 2, (3063,3501)),
    ('bruges', 2, 4, (1000,1000)),
    ('chicago', 15, 20, (3000,3000)),
    ('christchurch', 73, 730, (1500,1500)),
    ('DC', 1, 1, (1600,1600)),
    ('khartoum', 345,667, (390,390)),
    ('kitsap', 15, 20, (3000,3000)),
    ('NewHaven', 1, 1, (3000,3000)),
    ('NewYork', 1, 2, (1500,1500)),
    ('Norfolk', 1, 1, (3000,3000)),
    ('paris', 391,757,(390,390)),
    ('potsdam', 10,14,(600,600)),
    ('rio', 2360, 4580, (438,406)),
    ('SanFrancisco', 1, 2, (3000,3000)),
    ('Seekonk', 1,2, (3000,3000)),
    ('shanghai', 1558,3024,(390,390)),
    ('toulouse', 2,2,(3504,3452)),
    ('tyrol-w', 15,20,(3000,3000)),
    ('vegas', 1310,2541,(390,390)),
    ('vienna', 15,20,(3000,3000))
]


class MiniworldSup(BaseSupervisedDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        sup_train_datasets = []
        val_datasets = []
        for city_info in city_info_list:

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
                MiniworldCity(
                    data_path=os.path.join(self.data_dir, city_info[0]),
                    labeled_image_paths=labeled_image_paths,
                    label_paths=label_paths,
                    unlabeled_image_paths=unlabeled_image_paths,
                    image_size=city_info[3],
                    idxs=labeled_idxs[city_info[1]:],
                    crop=128
                )
            )
            val_datasets.append(
                MiniworldCity(
                    data_path=os.path.join(self.data_dir, city_info[0]),
                    labeled_image_paths=labeled_image_paths,
                    label_paths=label_paths,
                    unlabeled_image_paths=unlabeled_image_paths,
                    image_size=city_info[3],
                    idxs=labeled_idxs[:city_info[1]],
                    crop=128
                )
            )

        self.sup_train_set = ConcatDataset(sup_train_datasets)
        self.val_set = ConcatDataset(val_datasets)