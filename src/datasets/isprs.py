from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from albumentations import RandomCrop
from transforms import Merge_labels
import rasterio
from rasterio.windows import Window
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

class Isprs(Dataset):

    def __init__(self, data_path, idxs, crop, transforms):

        super(Isprs, self).__init__()
        self.data_path = os.path.join(data_path,'ISPRS_VAIHINGEN')
        self.idxs = idxs
        self.crop = crop
        self.transforms = transforms

    def get_crop_window(self, dataset):

        cols = dataset.width
        rows = dataset.height
        cx = np.random.randint(0, cols - self.crop - 1)
        cy = np.random.randint(0, rows - self.crop - 1)
        w = Window(cx, cy, self.crop, self.crop)

        return w

    def get_image(self, idx):

        # Surface model
        dsm_filepath = os.path.join(self.data_path, 'dsm',
                                    'dsm_09cm_matching_area{}.tif'.format(idx))
        # True orthophoto
        top_filepath = os.path.join(self.data_path, 'top',
                                    'top_mosaic_09cm_area{}.tif'.format(idx))

        with rasterio.open(dsm_filepath) as dsm_dataset:

            window = self.get_crop_window(dsm_dataset)
            dsm = dsm_dataset.read(
                window=window, out_dtype=np.float32
            ).transpose(1,2,0) / 255

        with rasterio.open(top_filepath) as top_dataset:

            top = top_dataset.read(
                window=window, out_dtype=np.float32
            ).transpose(1,2,0) / 255

        input = np.concatenate((top, dsm), axis=2)

        return input, window

    def get_truth(self, idx, window):

        # Ground truth
        gt_filepath = os.path.join(self.data_path, 'gts_for_participants',
                                   'top_mosaic_09cm_area{}.tif'.format(idx))

        with rasterio.open(gt_filepath) as gt_dataset:

            gt = gt_dataset.read(
                window=window
            ).transpose(1,2,0)

        return gt

    def isprs_colors_to_labels(self, data):

        """Convert colors to labels."""
        labels = np.zeros(data.shape[:2], dtype=int)

        colors = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [255, 255, 0],
                  [0, 255, 0], [255, 0, 0]]

        for id_col, col in enumerate(colors):
            d = (data[:, :, 0] == col[0])
            d = np.logical_and(d, (data[:, :, 1] == col[1]))
            d = np.logical_and(d, (data[:, :, 2] == col[2]))
            labels[d] = id_col

        return labels

    def __len__(self):

        return len(self.idxs)

    def __getitem__(self, idx):

        raise NotImplementedError


class Isprs_unlabeled(Isprs):

    def __init__(self,
                 data_path,
                 idxs,
                 crop,
                 transforms=None):

        super(Isprs_unlabeled, self).__init__(data_path, idxs, crop,
                                              transforms)


    def __getitem__(self, idx):

        idx = self.idxs[idx]
        image, window = self.get_image(idx)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        return image

class Isprs_labeled(Isprs):

    def __init__(self,
                 data_path,
                 idxs,
                 crop,
                 transforms=None):

        super(Isprs_labeled, self).__init__(data_path, idxs, crop, transforms)
        self.label_merger = Merge_labels([[0], [1]])

    def __getitem__(self, idx):

        idx = self.idxs[idx]
        image, window = self.get_image(idx)
        ground_truth = self.get_truth(idx, window)
        ground_truth = self.isprs_colors_to_labels(ground_truth)
        ground_truth = self.label_merger(ground_truth)

        if self.transforms is not None:
            transformed = self.transforms(image=image,
                                          mask=ground_truth)
            image = transformed['image']
            ground_truth = transformed['mask']

        # if self.co_transforms is not None:
        #     images, ground_truth = self.co_transforms(images, ground_truth)
        # if self.image_transforms is not None:
        #     images = self.image_transforms(images)
        # if self.label_transforms is not None:
        #     ground_truth = self.label_transforms(ground_truth)

        return image, ground_truth



