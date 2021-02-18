from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from albumentations import RandomCrop
from src.transforms import Merge_labels

class Isprs(Dataset):

    def __init__(self, data_path, idxs):

        super(Isprs, self).__init__()
        self.data_path = data_path
        self.idxs = idxs

    def get_images(self, idx):

        # Surface model
        dsm_filepath = os.path.join(self.data_path, 'dsm',
                                    'dsm_09cm_matching_area{}.tif'.format(idx))
        # True orthophoto
        top_filepath = os.path.join(self.data_path, 'top',
                                    'top_mosaic_09cm_area{}.tif'.format(idx))

        top = np.array(Image.open(top_filepath), dtype=np.float32) / 255
        dsm = np.array(Image.open(dsm_filepath), dtype=np.float32) / 255
        input = np.concatenate((top, np.expand_dims(dsm, axis=2)), axis=2)

        return input

    def get_truth(self, idx):

        # Ground truth
        gt_filepath = os.path.join(self.data_path, 'gts_for_participants',
                                   'top_mosaic_09cm_area{}.tif'.format(idx))
        gt = np.array(Image.open(gt_filepath))

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
                 transforms=None):

        super(Isprs_unlabeled, self).__init__(data_path, idxs)
        self.transforms = transforms

    def __getitem__(self, idx):

        idx = self.idxs[idx]
        images = self.get_images(idx)
        if self.transforms is not None:
            images = self.transforms(image=images)['image']

        return images

class Isprs_labeled(Isprs):

    def __init__(self,
                 data_path,
                 idxs,
                 transforms=None):

        super(Isprs_labeled, self).__init__(data_path, idxs)
        self.transforms = transforms
        self.label_merger = Merge_labels([[0], [1]])

    def __getitem__(self, idx):

        idx = self.idxs[idx]
        images = self.get_images(idx)
        ground_truth = self.get_truth(idx)
        ground_truth = self.isprs_colors_to_labels(ground_truth)
        ground_truth = self.label_merger(ground_truth)

        if self.transforms is not None:
            transformed = self.transforms(image=images,
                                          mask=ground_truth)
            images = transformed['image']
            ground_truth = transformed['mask']

        # if self.co_transforms is not None:
        #     images, ground_truth = self.co_transforms(images, ground_truth)
        # if self.image_transforms is not None:
        #     images = self.image_transforms(images)
        # if self.label_transforms is not None:
        #     ground_truth = self.label_transforms(ground_truth)

        return images, ground_truth



