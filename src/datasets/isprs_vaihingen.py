import os
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class IsprsVaihingen(Dataset):

    # The labeled and unlabeled image indices are properties of the class
    # independent of its instanciation.
    labeled_image_paths = [
        'top/top_mosaic_09cm_area{}.tif'.format(i) for i in [
            1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
            34, 37
        ]
    ]
    unlabeled_image_paths = [
        'top/top_mosaic_09cm_area{}.tif'.format(i) for i in [
            2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29,
            31, 33, 35, 38
        ]
    ]
    label_paths = [
        'gts_for_participants/top_mosaic_09cm_area{}.tif'.format(i) for i in [
            1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
            34, 37
        ]
    ]

    mean_labeled_pixels = [0.4727, 0.3205, 0.3159]
    std_labeled_pixels = [0.2100, 0.1496, 0.1426]

    @classmethod
    def colors_to_labels(cls, labels_color):
        labels = np.zeros(labels_color.shape[:2], dtype=int)

        colors = [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [255, 255, 0],
            [0, 255, 0],
            [255, 0, 0],
        ]

        for id_col, col in enumerate(colors):
            d = labels_color[:, :, 0] == col[0]
            d = np.logical_and(d, (labels_color[:, :, 1] == col[1]))
            d = np.logical_and(d, (labels_color[:, :, 2] == col[2]))
            labels[d] = id_col

        return labels


    def __init__(self, data_path, idxs, crop):

        super(IsprsVaihingen, self).__init__()
        self.data_path = data_path
        self.idxs = idxs
        self.crop = crop
        self.approx_crop_per_image = int(
            self.__image_size__ / (crop**2)
        )
        self.normalize = A.Normalize(
            mean=self.mean_labeled_pixels,
            std=self.std_labeled_pixels
        )


    def get_crop_window(self, image_file):

        cols = image_file.width
        rows = image_file.height
        cx = np.random.randint(0, cols - self.crop - 1)
        cy = np.random.randint(0, rows - self.crop - 1)
        w = Window(cx, cy, self.crop, self.crop)

        return w

    def get_image(self, idx):

        # True orthophoto
        top_filepath = os.path.join(
            self.data_path, self.labeled_image_paths[idx]
        )

        with rasterio.open(top_filepath) as image_file:

            window = self.get_crop_window(image_file)
            top = image_file.read(window=window, out_dtype=np.float32)
            top = top.transpose(1, 2, 0) / 255

        # If we want to use the surface model
        # dsm_filepath = os.path.join(self.data_path, 'dsm',
        #                             'dsm_09cm_matching_area{}.tif'.format(idx))
        # with rasterio.open(dsm_filepath) as dsm_dataset:
        #
        #     dsm = dsm_dataset.read(
        #         window=window, out_dtype=np.float32
        #     ).transpose(1,2,0) / 255
        # image = np.concatenate((top, dsm), axis=2)

        image = top

        return image, window

    def get_label(self, idx, window):

        # Ground truth
        label_filepath = os.path.join(
            self.data_path,
            self.label_paths[idx],
        )

        with rasterio.open(label_filepath) as label_file:

            gt = label_file.read(window=window).transpose(1, 2, 0)

        return gt

    def __len__(self):

        return self.approx_crop_per_image * len(self.idxs)

    def __getitem__(self, idx):

        raise NotImplementedError

    # The length of the dataset should be the number of get_item calls needed to
    # span the whole dataset. If get_item gives the full image, this is obviously
    # the total number of images in the dataset.
    # On the contrary, here get_item only gives a cropped tile from the image. Given the
    # crop parameter of the class and the average image size, provided they are all
    # close, we can say approx how many get_item calls are needed.
    @property
    def __image_size__(cls):

        return 1900 * 2600

class IsprsVaihingenUnlabeled(IsprsVaihingen):

    def __init__(self, data_path, idxs, crop):

        super(IsprsVaihingenUnlabeled, self).__init__(data_path, idxs, crop)

    def __getitem__(self, idx):

        idx = self.idxs[idx % len(self.idxs)]
        image, window = self.get_image(idx)

        return image


class IsprsVaihingenLabeled(IsprsVaihingen):

    def __init__(self, data_path, idxs, crop):

        super(IsprsVaihingenLabeled, self).__init__(data_path, idxs, crop)

    def __getitem__(self, idx):

        idx = self.idxs[idx % len(self.idxs)]
        image, window = self.get_image(idx)
        label_colors = self.get_label(idx, window)
        label = self.colors_to_labels(label_colors)

        return image, label
