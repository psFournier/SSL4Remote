import os
from torch.utils.data import Dataset
import torch
from dl_toolbox.utils import get_tiles
import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window
from dl_toolbox.utils import MergeLabels, OneHot
from dl_toolbox.torch_datasets.utils import *

class RasterDs(Dataset):

    labels = {}
    stats = {}

    def __init__(
            self,
            image_path,
            tile,
            fixed_crops,
            crop_size,
            crop_step,
            img_aug,
            label_path=None,
            one_hot=True,
            *args,
            **kwargs
            ):

        self.image_path = image_path
        self.label_path = label_path
        self.tile = tile
        self.crop_windows = list(get_tiles(
            nols=tile.width, 
            nrows=tile.height, 
            size=crop_size, 
            step=crop_step,
            row_offset=tile.row_off, 
            col_offset=tile.col_off)) if fixed_crops else None
        self.crop_size = crop_size
        self.img_aug = get_transforms(img_aug)

       # self.merge_labels = merge_labels
       # if merge_labels is None:
       #     self.labels, self.label_names = map(list, zip(*self.DATASET_DESC['labels']))
       #     self.label_merger = None
       # else:
       #     labels, self.label_names = merge_labels
       #     self.labels = list(range(len(labels)))
       #     self.label_merger = MergeLabels(labels)

        self.one_hot = OneHot(list(range(len(self.labels)))) if one_hot else None

    def read_label(self, label_path, window):
        pass

    def read_image(self, image_path, window):
        pass

    def __len__(self):

        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        
        if self.crop_windows:
            window = self.crop_windows[idx]
        else:
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)

        image = self.read_image(
            image_path=self.image_path,
            window=window
        )
        image = torch.from_numpy(image).float().contiguous()
       
        label = None
        if self.label_path:
            label = self.read_label(
                label_path=self.label_path,
                window=window
            )
            if self.one_hot: label = self.one_hot(label)
            label = torch.from_numpy(label).float().contiguous()

        if self.img_aug is not None:
            end_image, end_mask = self.img_aug(img=image, label=label)
        else:
            end_image, end_mask = image, label

        return {'orig_image':image,
                'orig_mask':label,
                'image':end_image,
                'window':window,
                'mask':end_mask}

    @classmethod
    def labels_to_rgb(cls, labels):

        # Inputs shape : B,H,W or H,W
        # Outputs shape : B,H,W,3 or H,W,3
        
        rgb = np.zeros(shape=(*labels.shape, 3), dtype=np.uint8)
        for label, key in enumerate(cls.labels):
            mask = np.array(labels == label)
            rgb[mask] = np.array(cls.labels[key]['color'])

        return rgb

    @classmethod
    def rgb_to_labels(cls, rgb):

        # Inputs shape : B,H,W,3 or H,W,3
        # Outputs shape : B,H,W or H,W

        labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
        for label, key in enumerate(cls.labels):
            c = cls.labels[key]['color']
            d = rgb[..., 0] == c[0]
            d = np.logical_and(d, (rgb[..., 1] == c[1]))
            d = np.logical_and(d, (rgb[..., 2] == c[2]))
            labels[d] = label

        return labels
    
#    @classmethod
#    def raw_labels_to_labels(cls, labels):
#
#
#
#        if dataset=='semcity':
#            return rgb_to_labels(labels, dataset=dataset)
#        elif dataset=='digitanie':
#            return torch.squeeze(torch.from_numpy(labels)).long()
#        else:
#            raise NotImplementedError

def main():

    dataset = DigitanieDs(
        image_path='/d/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7_img_normalized.tif',
        label_path='/d/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7.tif',
        crop_size=256,
        crop_step=256,
        img_aug='no',
        tile=Window(col_off=500, row_off=502, width=400, height=400),
        fixed_crops=False
    )

    for data in dataset:
        pass
    img = plt.imshow(dataset[0]['image'].numpy().transpose(1,2,0))

    plt.show()


if __name__ == '__main__':
    main()
