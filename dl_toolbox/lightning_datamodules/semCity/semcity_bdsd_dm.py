from lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from torch_datasets import SemcityBdsdDs
import os
import numpy as np
from torch.utils.data import ConcatDataset


class SemcityBdsdDm(BaseSupervisedDatamodule):

    def __init__(self, image_path, label_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path
        self.label_path = label_path

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--image_path", type=str)
        parser.add_argument("--label_path", type=str, default=None)
        return parser

    def setup(self, stage=None):
        self.sup_train_set = SemcityBdsdDs(
            image_path=self.image_path,
            label_path=self.label_path,
            tile_size=(863,876),
            tile_step=(863,876),
            crop_size=self.crop_size,
            img_aug=self.img_aug
        )
        self.sup_train_set.idxs = self.sup_train_set.idxs[::3] + self.sup_train_set.idxs[1::3]
        self.val_set = SemcityBdsdDs(
            image_path=self.image_path,
            label_path=self.label_path,
            tile_size=(863,876),
            tile_step=(863,876),
            crop_size=self.crop_size,
            img_aug=self.img_aug
        )
        self.val_set.idxs = self.val_set.idxs[2::3]

    @property
    def class_names(self):
        return [label[2] for label in self.sup_train_set.labels_desc]

    def label_to_rgb(self, labels):

        rgb_label = np.zeros(shape=(*labels.shape, 3), dtype=float)
        for val, color, _, _ in self.sup_train_set.labels_desc:
            mask = np.array(labels == val)
            rgb_label[mask] = np.array(color)
        rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

        return rgb_label

class SemcityBdsdDmSemisup(SemcityBdsdDm, BaseSemisupDatamodule):

    def __init__(self, data_dir, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument('--data_dir', type=str)

        return parser

    def setup(self, stage=None):

        super(SemcityBdsdDmSemisup, self).setup(stage=stage)
        nums = ['09','12','06','01','05','11','10','02','14','15','13','16']
        image_paths = [f'{self.data_dir}/TLS_BDSD_M_{num}.tif' for num in nums]
        unsup_train_sets = []
        for image_path in image_paths:
            set = SemcityBdsdDs(
                image_path=image_path,
                tile_size=(863, 876),
                tile_step=(863, 876),
                crop_size=self.unsup_crop_size
            )
            unsup_train_sets.append(set)
        self.unsup_train_set = ConcatDataset(unsup_train_sets)

