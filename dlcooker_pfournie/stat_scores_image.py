import rasterio as rio
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--pred_path", type=str)
parser.add_argument("--label_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--class_id", type=int)
args = parser.parse_args()

label = rio.open(args.label_path).read(out_dtype=np.uint8) // 255
with rio.open(args.pred_path) as pred_file:
    pred_profile = pred_file.profile
    pred = pred_file.read(out_dtype=np.uint8)
    height, width = pred_profile['height'], pred_profile['width']

label_bool = np.squeeze(label == args.class_id)
pred_bool = np.squeeze(pred == args.class_id)
overlay = np.zeros(shape=(height, width, 3), dtype=np.uint8)

# Correct predictions (Hits) painted with green
overlay[label_bool & pred_bool] = np.array([0, 250, 0], dtype=overlay.dtype)
# Misses painted with red
overlay[label_bool & ~pred_bool] = np.array([250, 0, 0], dtype=overlay.dtype)
# False alarm painted with yellow
overlay[~label_bool & pred_bool] = np.array([250, 250, 0], dtype=overlay.dtype)

pred_profile.update(count=3)
pred_profile.update(nodata=None)
with rio.open(args.output_path, 'w', **pred_profile) as dst:
    dst.write(overlay.transpose(2, 0, 1))