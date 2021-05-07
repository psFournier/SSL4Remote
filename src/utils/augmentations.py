import albumentations as A
from typing import Tuple, List
import numpy as np
from utils import Mixup


# __all__ = [
#     "D4_augmentations",
#     "medium_augmentations",
#     "hard_augmentations",
#     "get_augmentations",
# ]




def get_augment(names, always_apply=False):

    d = {'always_apply': always_apply}

    augments = {
        "d4": [A.RandomRotate90(p=1, **d),
               A.HorizontalFlip(p=0.5, **d),
               A.VerticalFlip(p=0.5, **d),
               A.Transpose(p=0.5, **d)],
        "clahe": [A.CLAHE(**d)],
        "sharpen": [A.IAASharpen(**d)],
        "blur": [A.GaussianBlur(**d)],
        "gamma": [A.RandomGamma(**d)],
        "hsv": [A.HueSaturationValue(**d)],
        "contrast": [A.RandomBrightnessContrast(brightness_by_max=True, **d)],
        "allcolor": [A.CLAHE(**d),
                     A.RandomGamma(**d),
                     A.HueSaturationValue(**d),
                     A.RandomBrightnessContrast(brightness_by_max=True,**d)],
        "allgeometric":[A.IAASharpen(**d),
                        A.GaussianBlur(**d)],
        "maskdrop": [A.MaskDropout(max_objects=2, mask_fill_value=0, **d)]
    }

    l = sum([augments[name] for name in names], [])

    return l

batch_augments = {
    "mixup": [Mixup(alpha=0.4)],
}

def get_batch_augment(names):

    l = sum([batch_augments[name] for name in names], [])

    return l


# def get_augmentations(augmentation):
#     if augmentation == "hard":
#         aug_transform = hard_augmentations()
#     elif augmentation == "medium":
#         aug_transform = medium_augmentations()
#     elif augmentation == "d4":
#         aug_transform = D4_augmentations()
#     # elif augmentation == "mixup":
#     #     aug_transform =
#     elif augmentation == "dropout":
#         aug_transform = dropout()
#     else:
#         aug_transform = []
#
#     return aug_transform

class MergeLabels:
    def __init__(self, labels):

        self.labels = labels

    def __call__(self, L):
        """
        If self.labels is [[0,1,2],[3,4,5]], then all pixels in classes [0,1,2]
        will be set to label 0 and all pixels in classes [3,4,5] will be set to
        label 1.
        :param L:
        :return:
        """
        ret = np.zeros(L.shape, dtype=L.dtype)
        for i, lab in enumerate(self.labels):
            for j in lab:
                ret[L == j] = i

        return ret





def dropout():
    return [A.OneOf(
        [
            A.CoarseDropout(),
            A.MaskDropout(max_objects=2, mask_fill_value=0)
        ]
    )]


def medium_augmentations(mask_dropout=True):
    return [
        A.RandomRotate90(p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15),
        # Add occasion blur/sharpening
        A.OneOf([A.GaussianBlur(), A.IAASharpen(), A.NoOp()]),
        # Spatial-preserving augmentations:
        # A.OneOf([A.CoarseDropout(), A.MaskDropout(max_objects=5) if
        # mask_dropout  else A.NoOp(), A.NoOp()]),
        # A.GaussNoise(),
        A.OneOf([A.RandomBrightnessContrast(), A.CLAHE(),
                 A.HueSaturationValue(), A.RGBShift(), A.RandomGamma()]),
        # Weather effects
        # A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
    ]


def hard_augmentations(mask_dropout=True) -> List[A.DualTransform]:
    return [
        # D4 Augmentations
        A.RandomRotate90(p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        # Spatial augmentations
        # A.OneOf(
        #     [
        #         A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_REFLECT101),
        #         A.ElasticTransform(border_mode=cv2.BORDER_REFLECT101, alpha_affine=5),
        #     ]
        # ),
        # Color augmentations
        A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_by_max=True),
                A.CLAHE(),
                A.FancyPCA(),
                A.HueSaturationValue(),
                A.RGBShift(),
                A.RandomGamma(),
            ]
        ),
        # Dropout & Shuffle
        A.OneOf(
            [
                # A.RandomGridShuffle(),
                A.CoarseDropout(),
                A.MaskDropout(max_objects=2, mask_fill_value=0) if mask_dropout else A.NoOp(),
            ]
        ),
        # Add occasion blur
        A.OneOf([A.GaussianBlur(), A.GaussNoise(), A.IAAAdditiveGaussianNoise()]),
        # Weather effects
        # A.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
    ]


