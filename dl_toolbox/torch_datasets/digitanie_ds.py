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

class DigitanieDs(RasterDs):

    labels = {
        0: {
            'name':'other',
            'color':(0,0,0)
        },

    }

    def __init__(self,
                 *args, 
                 **kwargs
    ):

        self.DATASET_DESC['labels'] = [
            (0, 'other'),
            (1, 'bare ground'),
            (2, 'low vegetation'),
            (3, 'water'),
            (4, 'building'),
            (5, 'high vegetation'),
            (6, 'parking'),
            (7, 'pedestrian'),
            (8, 'road'),
            (9, 'railways'),
            (10, 'swimmingpool')
        ]
        self.DATASET_DESC['label_colors'] = [
            (0,0,0),
            (100,50,0),
            (0,250,50),
            (0,50,250),
            (250,50,50),
            (0,100,50),
            (200,200,200),
            (200,150,50),
            (100,100,100),
            (200,100,200),
            (50,150,250)
        ]
        
        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels(
            [[i] for i in self.labels.keys()]
        )


class DigitanieToulouseDs(DigitanieDs):

    stats = DigitanieDs.stats
    stats['min'] = np.array([0, 0.0029, 0.0028, 0])
    stats['max'] = np.array([1.5431, 1.1549, 1.1198, 2.0693])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.full_raster_path = full_raster_path

    def read_image(self, img_path, window):

        image = read_window_from_big_raster(
            window=window,
            path=img_path,
            raster_path=self.full_raster_path
        )

        image = minmax(
            image, 
            self.stats['min'],
            self.stats['max']
        )

        return image

    def read_label(self, label_path, window):
    
        label = read_window_basic(
            window=window,
            path=label_path
        )
        label = self.label_merger(label)

        return label





class DigitanieParisDs(DigitanieDs):

    def __init__(self, *args, **kwargs):

        self.DATASET_DESC['min'] = np.array([0, 0, 0, 0])
        self.DATASET_DESC['max'] = np.array([19051, 16216, 15239, 29244])
        super().__init__(*args, **kwargs)

class DigitanieMontpellierDs(DigitanieDs):

    def __init__(self, *args, **kwargs):

        self.DATASET_DESC['min'] = np.array([1,1,1,2])
        self.DATASET_DESC['max'] = np.array([4911,4736,4753,5586])
        super().__init__(*args, **kwargs)

class DigitanieBiarritzDs(DigitanieDs):

    def __init__(self, *args, **kwargs):

        self.DATASET_DESC['min'] = np.array([-544, -503, 473, -652])
        self.DATASET_DESC['max'] = np.array([19498, 19829, 17822, 27880])
        super().__init__(*args, **kwargs)

class DigitanieStrasbourgDs(DigitanieDs):

    def __init__(self, *args, **kwargs):

        self.DATASET_DESC['min'] = np.array([89, 159, 202,92])
        self.DATASET_DESC['max'] = np.array([4670, 4311, 4198, 6188])
        super().__init__(*args, **kwargs)


def main():
    
    image_path = '/d/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7_img_normalized.tif'
    dataset = DigitanieToulouseDs(
        image_path=image_path,
        label_path='/d/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7.tif',
        crop_size=256,
        crop_step=256,
        img_aug='no',
        tile=Window(col_off=500, row_off=502, width=400, height=400),
        read_window_fn=partial(
            read_window_basic,
            path=image_path
        ),
        norm_fn=partial(
            minmax,
            m=dataset.DATASET_DESC['min'],
            M=dataset.DATASET_DESC['max']
        ),
        fixed_crops=False
    )

    for data in dataset:
        pass
    img = plt.imshow(dataset[0]['image'][:3,...].numpy().transpose(1,2,0))

    plt.show()


if __name__ == '__main__':
    main()
