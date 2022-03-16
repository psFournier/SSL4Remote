import csv
import imagesize
import os
from dl_toolbox.utils import get_tiles

with open('/d/pfournie/ai4geo/data/SemcityTLS_DL/split.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    #path = '/work/OT/ai4geo/DATA/DATASETS/DIGITANIE'
    path = '/d/pfournie/ai4geo/data/SemcityTLS_DL'
    writer.writerow(['city',
                     'img_path',
                     'label_path',
                     'x0',
                     'y0',
                     'patch_width',
                     'patch_height',
                     'fold_id'
                     ])
    image_path = os.path.join(path, 'BDSD_M_3_4_7_8.tif')
    label_path = os.path.join(path, 'GT_3_4_7_8.tif')
    w, h = imagesize.get(image_path)
    for i, tile in enumerate(get_tiles(w, h, size=876, size2=863)):
        writer.writerow([
            'Toulouse',
            image_path,
            label_path,
            tile.col_off,
            tile.row_off,
            tile.width,
            tile.height,
            i % 5
        ])
