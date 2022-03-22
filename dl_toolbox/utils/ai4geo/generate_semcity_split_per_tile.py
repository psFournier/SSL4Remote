import csv
import imagesize
import os
from dl_toolbox.utils import get_tiles
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--output_file", type=str)
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()

with open(args.output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    path = args.data_dir
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
            0,
            image_path,
            label_path,
            tile.col_off,
            tile.row_off,
            tile.width,
            tile.height,
            i % 5
        ])
