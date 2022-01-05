from lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from torch_datasets import OneImage
import glob
from torch.utils.data import ConcatDataset
import numpy as np
from functools import partial
from torch_datasets import PhrPanNdviDs

class PhrPanNdviDm(BaseSupervisedDatamodule):

    def __init__(self, image_path, ndvi_path, label_path, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.image_path = image_path
        self.ndvi_path = ndvi_path
        self.label_path = label_path

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--image_path", type=str)
        parser.add_argument("--ndvi_path", type=str)
        parser.add_argument("--label_path", type=str, default=None)

        return parser

    def setup(self, stage=None):

        sup_train_set = PhrPanNdviDs(
            image_path=self.image_path,
            ndvi_path=self.ndvi_path,
            label_path=self.label_path,
            tile_size=2300,
            tile_step=2300,
            crop_size=self.crop_size
        )
        sup_train_set.idxs = sup_train_set.idxs[::3] + sup_train_set.idxs[1::3]
        val_set = PhrPanNdviDs(
            image_path=self.image_path,
            ndvi_path=self.ndvi_path,
            label_path=self.label_path,
            tile_size=2300,
            tile_step=2300,
            crop_size=self.crop_size
        )
        val_set.idxs = val_set.idxs[2::3]


class PhrPanNdviDmSemisup(PhrPanNdviDm, BaseSemisupDatamodule):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        super(PhrPanNdviDmSemisup, self).setup(stage=stage)

        self.unsup_train_set = PhrPanNdviDs(
            image_path=self.image_path,
            ndvi_path=self.ndvi_path,
            tile_size=2300,
            tile_step=2300,
            crop_size=self.crop_size
        )
