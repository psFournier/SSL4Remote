import os
import warnings

import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def isprs_colors_to_labels(data):
    labels = np.zeros(data.shape[:2], dtype=int)

    colors = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [0, 255, 0],
        [255, 0, 0],
    ]

    for id_col, col in enumerate(colors):
        d = data[:, :, 0] == col[0]
        d = np.logical_and(d, (data[:, :, 1] == col[1]))
        d = np.logical_and(d, (data[:, :, 2] == col[2]))
        labels[d] = id_col

    return labels


class IsprsVaihingen(Dataset):

    # The labeled and unlabeled image indices are properties of the class
    # independent of its instanciation.
    labeled_idxs = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32,
                    34, 37]
    unlabeled_idxs = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29,
                      31, 33, 35, 38]
    mean_labeled_pixels = [0.4727, 0.3205, 0.3159]
    std_labeled_pixels = [0.2100, 0.1496, 0.1426]

    def __init__(self, data_path, idxs, crop):

        super(IsprsVaihingen, self).__init__()
        self.data_path = os.path.join(data_path, "ISPRS_VAIHINGEN")
        self.idxs = idxs
        self.crop = crop

    def get_crop_window(self, dataset):

        cols = dataset.width
        rows = dataset.height
        cx = np.random.randint(0, cols - self.crop - 1)
        cy = np.random.randint(0, rows - self.crop - 1)
        w = Window(cx, cy, self.crop, self.crop)

        return w

    def get_image(self, idx):

        # True orthophoto
        top_filepath = os.path.join(
            self.data_path, "top", "top_mosaic_09cm_area{}.tif".format(idx)
        )

        with rasterio.open(top_filepath) as top_dataset:

            window = self.get_crop_window(top_dataset)
            top = top_dataset.read(window=window, out_dtype=np.float32)
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

    def get_truth(self, idx, window):

        # Ground truth
        gt_filepath = os.path.join(
            self.data_path,
            "gts_for_participants",
            "top_mosaic_09cm_area{}.tif".format(idx),
        )

        with rasterio.open(gt_filepath) as gt_dataset:

            gt = gt_dataset.read(window=window).transpose(1, 2, 0)

        return gt

    def __len__(self):

        return len(self.idxs)

    def __getitem__(self, idx):

        raise NotImplementedError


class IsprsVaihingenUnlabeled(IsprsVaihingen):
    def __init__(self, data_path, idxs, crop, transforms=None):

        super(IsprsVaihingenUnlabeled, self).__init__(data_path, idxs, crop)

    def __getitem__(self, idx):

        idx = self.idxs[idx]
        image, window = self.get_image(idx)

        return image


class IsprsVaihingenLabeled(IsprsVaihingen):
    def __init__(self, data_path, idxs, crop, transforms=None):

        super(IsprsVaihingenLabeled, self).__init__(data_path, idxs, crop)

    def __getitem__(self, idx):

        idx = self.idxs[idx]
        image, window = self.get_image(idx)
        ground_truth = self.get_truth(idx, window)
        ground_truth = isprs_colors_to_labels(ground_truth)

        return image, ground_truth
