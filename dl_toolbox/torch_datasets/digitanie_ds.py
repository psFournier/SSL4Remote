import os
from torch.utils.data import Dataset
import torch
from dl_toolbox.torch_datasets.commons import minmax
from dl_toolbox.utils import get_tiles
import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds
from dl_toolbox.utils import MergeLabels, OneHot
import matplotlib.pyplot as plt
from dl_toolbox.torch_datasets import RasterDs
from dl_toolbox.torch_datasets.utils import *
from functools import partial
from enum import Enum

digitanie_labels = {
    
    'base' : {
        'other': {'color': (0, 0, 0)},
        'bare ground': {'color': (100, 50, 0)},
        'low vegetation': {'color':(0, 250, 50)},
        'water': {'color': (0, 50, 250)},
        'building': {'color': (250, 50, 50)},
        'high vegetation': {'color': (0, 100, 50)},
        'parking': {'color': (200, 200, 200)},
        'pedestrian': {'color': (200, 150, 50)},
        'road': {'color': (100, 100, 100)},
        'railways': {'color': (200, 100, 200)},
        'swimming pool': {'color': (50, 150, 250)}
    },
    '8class' : {
        'other': {'color': (0, 0, 0)},
        'low vegetation': {'color':(0, 250, 50)},
        'water': {'color': (0, 50, 250)},
        'building': {'color': (250, 50, 50)},
        'high vegetation': {'color': (0, 100, 50)},
        'bitumen': {'color': (100, 100, 100)},
        'railways': {'color': (200, 100, 200)},
        'swimming pool': {'color': (50, 150, 250)}
    },
    'semcity' : {
        'other': {'color': (255, 255, 255)},
        'pervious surface': {'color': (34, 139, 34)},
        'water': {'color': (0, 0, 238)},
        'building': {'color': (238, 118, 33)},
        'high vegetation': {'color': (0, 222, 137)},
        'impervious surface': {'color': (38, 38, 38)}
    },
    'building' : {
        'background': {'color': (0,0,0)},
        'building': {'color': (255, 255, 255)}
    }
}

digitanie_label_mergers = {
    'base' : [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
    '8class' : [[0, 1, 7], [2], [3], [4], [5], [6, 8], [9], [10]],
    'semcity' : [[0, 9], [1, 2], [3, 10], [4], [5], [6, 7, 8]],
    'building' : [[0,1,2,3,5,6,7,8,9,10],[4]]
}

class DigitanieOldDs(RasterDs):

    def __init__(self, labels, label_merger, *args, **kwargs):
 
        self.labels = digitanie_labels[labels]
        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels(digitanie_label_mergers[label_merger])

    def read_image(self, image_path, window):

        image = read_window_basic(
            window=window,
            path=image_path
        )

        image = image[:3,...]

        return image

    def read_label(self, label_path, window):
    
        label = read_window_basic(
            window=window,
            path=label_path
        )
        label = np.squeeze(label)
        label = self.label_merger(label)
        

        return label


class DigitanieDs(RasterDs):

    def __init__(self, labels, label_merger, full_raster_path, *args, **kwargs):
 
        self.labels = digitanie_labels[labels]
        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels(digitanie_label_mergers[label_merger])
        self.full_raster_path = full_raster_path

    def read_image(self, image_path, window):
        
        if self.full_raster_path == image_path:
            image = read_window_basic(
                window=window,
                path=image_path
            )
        else:
            image = read_window_from_big_raster(
                window=window,
                path=image_path,
                raster_path=self.full_raster_path
            )

        image = image[:3,...]

        image = minmax(
            image, 
            self.stats['min'][:3,...],
            self.stats['max'][:3,...]
        )

        return image

    def read_label(self, label_path, window):
    
        label = read_window_basic(
            window=window,
            path=label_path
        )
        label = np.squeeze(label)
        label = self.label_merger(label)
        

        return label

class DigitanieToulouseDs(DigitanieDs):

    stats = {}
    stats['min'] = np.array([0, 0.0029, 0.0028, 0])
    stats['max'] = np.array([1.5431, 1.1549, 1.1198, 2.0693])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class DigitanieParisDs(DigitanieDs):
    
    stats = {}
    stats['min'] = np.array([0, 0, 0, 0])
    stats['max'] = np.array([19051, 16216, 15239, 29244])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class DigitanieMontpellierDs(DigitanieDs):
    
    stats = {}
    stats['min'] = np.array([1,1,1,2])
    stats['max'] = np.array([4911,4736,4753,5586])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class DigitanieBiarritzDs(DigitanieDs):

    stats = {}
    stats['min'] = np.array([-544, -503, 473, -652])
    stats['max'] = np.array([19498, 19829, 17822, 27880])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class DigitanieStrasbourgDs(DigitanieDs):

    stats = {}
    stats['min'] = np.array([89, 159, 202,92])
    stats['max'] = np.array([4670, 4311, 4198, 6188])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


def main():
    
    image_path = '/home/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7_img_normalized.tif'
    label_path = '/home/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7.tif'
    full_raster_path = '/home/pfournie/ai4geo/data/DIGITANIE/Toulouse/normalized_mergedTO.tif'
    dataset = DigitanieToulouse2Ds(
        image_path=image_path,
        label_path=label_path,
        full_raster_path=full_raster_path,
        crop_size=1024,
        crop_step=1024,
        img_aug='no',
        tile=Window(col_off=0, row_off=0, width=1024, height=1024),
        fixed_crops=False,
        one_hot=False
    )
    img = dataset[0]['mask']
    print(img.shape)
    img = DigitanieToulouse2Ds.labels_to_rgb(img.numpy())
    #img = img.numpy().transpose((1,2,0))
    img = plt.imsave('digitanie_ds_test.jpg', img)



if __name__ == '__main__':
    main()
