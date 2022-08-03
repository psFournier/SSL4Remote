from lightning_datamodules import BaseSupervisedDatamodule, BaseSemisupDatamodule
from torch_datasets import *

# cities_labeled = {
#     'christchurch': ChristchurchLabeled,
#     'austin': AustinLabeled,
#     'chicago': ChicagoLabeled,
#     'kitsap': KitsapLabeled,
#     'tyrol-w': TyrolwLabeled,
#     'vienna': ViennaLabeled
# }
#
# cities_unlabeled = {
#     'christchurch': ChristchurchUnlabeled,
#     'austin': AustinUnlabeled,
#     'chicago': ChicagoUnlabeled,
#     'kitsap': KitsapUnlabeled,
#     'tyrol-w': TyrolwUnlabeled,
#     'vienna': ViennaUnlabeled
# }
#
#
# class MiniworldV1(BaseSupervisedDatamodule):
#
#     def __init__(self, city, *args, **kwargs):
#
#         super().__init__(*args, **kwargs)
#         self.city = city
#
#     @classmethod
#     def add_model_specific_args(cls, parent_parser):
#
#         parser = super().add_model_specific_args(parent_parser)
#         parser.add_argument("--city", type=str, default='christchurch',
#                             help="Which city to train on.")
#
#         return parser
#
#     def setup(self, stage=None):
#
#         self.sup_train_set = cities_labeled[self.city](
#             data_path=self.data_dir,
#             crop=self.crop_size,
#             crop_step=self.crop_size
#         )
#
#         self.val_set = cities_labeled[self.city](
#             data_path=self.data_dir,
#             crop=self.crop_size,
#             fixed_crop=True,
#             crop_step=self.crop_size
#         )
#
#         train, val = self.train_val
#         if train != 0 and val != 0:
#             nb_labeled_images = len(self.sup_train_set.labeled_image_paths)
#             labeled_idxs = list(range(nb_labeled_images))
#             self.sup_train_set.path_idxs = labeled_idxs[:train]
#             self.val_set.path_idxs = labeled_idxs[train:train+val]
#         else:
#             self.sup_train_set.path_idxs = list(self.train_idxs)
#             self.val_set.path_idxs = list(self.val_idxs)
#
# class MiniworldV1Semisup(MiniworldV1, BaseSemisupDatamodule):
#
#     def __init__(self, *args, **kwargs):
#
#         super().__init__(*args, **kwargs)
#
#     def setup(self, stage=None):
#
#         self.sup_train_set = cities_labeled[self.city](
#             data_path=self.data_dir,
#             crop=self.crop_size,
#             crop_step=self.crop_size
#         )
#
#         self.val_set = cities_labeled[self.city](
#             data_path=self.data_dir,
#             crop=self.crop_size,
#             crop_step=self.crop_size,
#             fixed_crop=True
#         )
#
#         train, val = self.train_val
#         if train != 0 and val != 0:
#             nb_labeled_images = len(self.sup_train_set.labeled_image_paths)
#             labeled_idxs = list(range(nb_labeled_images))
#             self.sup_train_set.path_idxs = labeled_idxs[:train]
#             self.val_set.path_idxs = labeled_idxs[train:train+val]
#         else:
#             self.sup_train_set.path_idxs = list(self.train_idxs)
#             self.val_set.path_idxs = list(self.val_idxs)
#
#         self.unsup_train_set = cities_unlabeled[self.city](
#             data_path = self.data_dir,
#             crop=self.crop_size,
#             crop_step=self.crop_size
#         )
#
#         if self.unsup_train != 0:
#             nb_unlabeled_images = len(self.unsup_train_set.unlabeled_image_paths)
#             unlabeled_idxs = list(range(nb_unlabeled_images))
#             self.unsup_train_set.path_idxs = unlabeled_idxs[:self.unsup_train]
#         else:
#             self.unsup_train_set.path_idxs = list(self.unsup_train_idxs)
