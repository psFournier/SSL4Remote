import os
from torch.utils.data import Dataset
import torch
from dl_toolbox.torch_datasets.commons import minmax
from dl_toolbox.utils import get_tiles
import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds
from dl_toolbox.torch_datasets.utils import *

class SemcityBdsdDs(Dataset):

    DATASET_DESC = {
        'labels' : [
            (0, 'void'),
            (1, 'impervious surface'),
            (2, 'building'),
            (3, 'pervious surface'),
            (4, 'high vegetation'),
            (5, 'car'),
            (6, 'water'),
            (7, 'sport venue')
        ],
        'min' : np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        'max' : np.array([2902,4174,4726,5196,4569,4653,5709,3939]),
        'label_colors' : [
            (255, 255, 255),
            (38, 38, 38),
            (238, 118, 33),
            (34, 139, 34),
            (0, 222, 137),
            (255, 0, 0),
            (0, 0, 238),
            (160, 30, 230)
        ]
    }
#    labels_desc = [
#        , 'void', 1335825),
#        impervious surface', 13109372),
#         'building', 9101418),
#        'pervious surface', 12857668),
#        'high vegetation', 8214402),
#        ar', 1015653),
#        ater', 923176),
#         'sport venues', 1825718)
#    ]
#    # Min and max are 1 and 99 percentiles
#    image_stats = {
#        'num_channels': 8,
#        'min' : np.array([245, 166, 167, 107, 42, 105, 60, 48]),
#        'max' : np.array([615, 681, 1008, 1087, 732, 1065, 1126, 1046])
#    }
    color_map = {k: v for k,v in enumerate(DATASET_DESC['label_colors'])}

    def __init__(
        self,
        image_path,
        tile,
        fixed_crops,
        crop_size,
        crop_step,
        img_aug,
        label_path=None,
        merge_labels=None,
        one_hot_labels=True,
        *args,
        **kwargs):

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
        
        #self.merge_labels = merge_labels
        #if merge_labels is None:
        #    self.labels, self.label_names = map(list, zip(*self.DATASET_DESC['labels']))
        #    self.label_merger = None
        #else:
        #    labels, self.label_names = merge_labels
        #    self.labels = list(range(len(labels)))
        #    self.label_merger = MergeLabels(labels)

        #self.one_hot_labels = one_hot_labels
        #if self.one_hot_labels:
        #    self.one_hot = OneHot(self.labels)


    def __len__(self):

        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        
        if self.crop_windows:
            window = self.crop_windows[idx]
        else:
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
        
        label = None
        if self.label_path:
            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
            onehot_masks = []
            for color in self.DATASET_DESC['label_colors']:
                d = label[0, :, :] == color[0]
                d = np.logical_and(d, (label[1, :, :] == color[1]))
                d = np.logical_and(d, (label[2, :, :] == color[2]))
                onehot_masks.append(d.astype(float))
            onehot = np.stack(onehot_masks, axis=0)
            label = torch.from_numpy(np.stack(onehot_masks, axis=0)).float().contiguous()
            
        with rasterio.open(self.image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)
        m, M = self.DATASET_DESC['min'][[3,2,1]], self.DATASET_DESC['max'][[3,2,1]]
        image = torch.from_numpy(minmax(image[[3,2,1],...], m, M)).float().contiguous()

        if self.img_aug is not None:
            end_image, end_mask = self.img_aug(img=image, label=label)
        else:
            end_image, end_mask = image, label

        return {'orig_image':image,
                'orig_mask':label,
                'image':end_image,
                'window':window,
                'mask':end_mask}


def main():

    dataset = SemcityBdsdDs(
        image_path='/d/pfournie/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif',
        label_path='/d/pfournie/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif',
        crop_size=128,
        crop_step=128,
        img_aug='d4_color-1',
        tile=Window(col_off=876, row_off=863, width=876, height=863),
        fixed_crops=True
    )

    for data in dataset:

        print(data['window'])

if __name__ == '__main__':
    
    main()
