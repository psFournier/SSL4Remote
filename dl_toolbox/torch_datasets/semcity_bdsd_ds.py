import os
from torch.utils.data import Dataset
import torch
from dl_toolbox.torch_datasets.commons import minmax
from dl_toolbox.utils import get_tiles
import rasterio
import imagesize
import numpy as np
import matplotlib.pyplot as plt
from rasterio.windows import Window, bounds, from_bounds
from dl_toolbox.torch_datasets import RasterDs
from dl_toolbox.torch_datasets.utils import *
from dl_toolbox.utils import MergeLabels, OneHot

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

BASE_LABELS = {
    'void': {'color': (255, 255, 255)},
    'impervious surface': {'color': (38, 38, 38)},
    'building': {'color': (238, 118, 33)},
    'pervious surface': {'color': (34, 139, 34)},
    'high vegetation': {'color': (0, 222, 137)},
    'car': {'color': (255, 0, 0)},
    'water': {'color': (0, 0, 238)},
    'sport venue': {'color': (160, 30, 230)}
}

NO_MERGE = [[0], [1], [2], [3], [4], [5], [6], [7]]

class SemcityBdsdDs(RasterDs):

    labels = BASE_LABELS
    stats = {
        'min': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        'max': np.array([2902,4174,4726,5196,4569,4653,5709,3939])
    }

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels(NO_MERGE)

    def read_image(self, image_path, window):

        image = read_window_basic(
            window=window,
            path=image_path
        )

        image = image[[3,2,1],...]

        image = minmax(
            image, 
            self.stats['min'][[3,2,1],...],
            self.stats['max'][[3,2,1],...]
        )

        return image

    def read_label(self, label_path, window):
    
        rgb = read_window_basic(
            window=window,
            path=label_path
        )
        label = self.rgb_to_labels(rgb.transpose((1,2,0)))
        label = self.label_merger(label)

        return label
           

def main():

    dataset = SemcityBdsdDs(
        image_path='/home/pfournie/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif',
        label_path='/home/pfournie/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif',
        crop_size=800,
        crop_step=800,
        img_aug='no',
        tile=Window(col_off=876, row_off=863, width=876, height=863),
        fixed_crops=False,
        one_hot=False
    )
    img = dataset[0]['mask']
    print(img.shape)
    img = SemcityBdsdDs.labels_to_rgb(img.numpy())
    #img = img.numpy().transpose((1,2,0))
    img = plt.imsave('semcity_ds_test.jpg', img)
    
if __name__ == '__main__':
    
    main()
