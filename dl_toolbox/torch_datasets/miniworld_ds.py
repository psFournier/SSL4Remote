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
from dl_toolbox.utils import LabelsToRGB

miniworld_labels = {

    'base_labels' : {
        'background': {'color': (0,0,0)},
        'building': {'color': (255, 255, 255)}
    }
}

miniworld_label_mergers = {
    'no_merge' : [[0], [1]],
}


class MiniworldDs(RasterDs):

    def __init__(self, labels, label_merger, *args, **kwargs):
 
        self.labels = miniworld_labels[labels]
        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels(miniworld_label_mergers[label_merger])

    def read_image(self, image_path, window):

        image = read_window_basic(
            window=window,
            path=image_path,
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

class MiniworldChristchurchDs(MiniworldDs):

    stats = {}
    stats['min'] = np.array([0, 0.0029, 0.0028, 0])
    stats['max'] = np.array([1.5431, 1.1549, 1.1198, 2.0693])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class MiniworldViennaDs(MiniworldDs):
    
    stats = {}
    stats['min'] = np.array([0, 0, 0, 0])
    stats['max'] = np.array([19051, 16216, 15239, 29244])

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


def main():
    
    image_path = '/home/pfournie/ai4geo/data/miniworld/Christchurch' \
                 '/toulouse_tuile_7_img_normalized.tif'
    label_path = '/home/pfournie/ai4geo/data/DIGITANIE/Toulouse/toulouse_tuile_7.tif'
    dataset = MiniworldChristchurchDs(
        image_path=image_path,
        label_path=label_path,
        crop_size=1024,
        crop_step=1024,
        img_aug='no',
        tile=Window(col_off=0, row_off=0, width=1024, height=1024),
        fixed_crops=False,
        one_hot=False
    )
    img = dataset[0]['mask']
    print(img.shape)
    img = LabelsToRGB(img.numpy())
    #img = img.numpy().transpose((1,2,0))
    img = plt.imsave('digitanie_ds_test.jpg', img)



if __name__ == '__main__':
    main()
