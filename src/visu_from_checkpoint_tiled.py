from pl_modules import SupervisedBaseline
import rasterio as rio
import numpy as np
from rasterio.windows import Window
import matplotlib.pyplot as plt
from torch_datasets import MiniworldCities
from transforms import MergeLabels
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from itertools import product
import torch

plt.switch_backend("TkAgg")


def visualize(image, predictions, truth, overlay):
    fontsize = 18
    f, ax = plt.subplots(2, 2, figsize=(8, 8))

    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    ax[1, 0].imshow(truth)
    ax[1, 0].set_title('Ground truth', fontsize=fontsize)

    ax[0, 1].imshow(predictions)
    ax[0, 1].set_title('Model predictions', fontsize=fontsize)

    ax[1, 1].imshow(overlay)
    ax[1, 1].set_title('Evaluation', fontsize=fontsize)

def get_tiles(ds, nols, nrows, width=256, height=256, col_step=128,
              row_step=128):
    assert col_step <= width
    assert row_step <= height
    assert nols <= ds.meta['width']
    assert nrows <= ds.meta['height']

    max_col_offset = int((nols-width)/col_step)+1
    # Remove all offsets such that offset+width > nols and add one offset to
    # reach nols
    col_offsets = list(range(0, nols, col_step))[:max_col_offset+1]
    col_offsets[max_col_offset] = nols - width

    max_row_offset = int((nrows - height) / row_step) + 1
    # Remove all offsets such that offset+width > nols and add one offset to
    # reach nols
    row_offsets = list(range(0, nrows, row_step))[:max_row_offset+1]
    row_offsets[max_row_offset] = nrows - height

    offsets = product(col_offsets, row_offsets)
    big_window = Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window = Window(col_off=col_off, row_off=row_off, width=width,
                                height=height).intersection(big_window)
        yield window


ckpt_path = '/home/pierre/PycharmProjects/RemoteSensing/outputs' \
    '/baseline_christchurch_noaug_2021-04-16/checkpoints/epoch=96-step=30360.ckpt'

module = SupervisedBaseline.load_from_checkpoint(ckpt_path)
module.eval()
augment = A.Compose([
    A.Normalize(),
    ToTensorV2(transpose_mask=False)
])

image_path = '/home/pierre/Documents/ONERA/ai4geo/airs/test/41_x.png'
label_path = '/home/pierre/Documents/ONERA/ai4geo/airs/test/41_y.png'


windows = []
tiles = []
with rio.open(image_path) as image_file:

    image = image_file.read(window=Window(col_off=0,
                                                 row_off=0,
                                                 width=300,
                                                 height=300),
                            out_dtype=np.uint8).transpose(1, 2, 0)

    for window in get_tiles(image_file, width=128, height=128, col_step=64,
                            row_step=64, nols=300, nrows=300):

        tile = image_file.read(window=window,
                               out_dtype=np.uint8).transpose(1, 2, 0)
        augmented = augment(image=tile)['image']
        tiles.append(augmented)
        windows.append(window)

batch = torch.stack(tiles, dim=0)
pred = module.network(batch)
pred = np.argmax(pred.detach().numpy().squeeze(), axis=1).astype(bool)
preds = [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]

stitched_pred = np.zeros(shape=(300,300))
nb_tiles = np.zeros(shape=(300, 300))
for pred, window in zip(preds, windows):
    stitched_pred[window.row_off:window.row_off+window.width,
    window.col_off:window.col_off+window.height] += pred
    nb_tiles[window.row_off:window.row_off+window.width,
    window.col_off:window.col_off+window.height] += 1
avg_pred = np.rint(np.divide(stitched_pred, nb_tiles)).astype(bool)


with rio.open(label_path) as label_file:
    label = label_file.read(window=Window(col_off=0,
                                          row_off=0,
                                          width=300,
                                          height=300),
                            out_dtype=np.uint8).transpose(1, 2, 0)

gt = MiniworldCities.colors_to_labels(label)

label_merger = MergeLabels([[0], [1]])
gt = label_merger(gt).astype(bool)


overlay = image.copy()
# Correct predictions (Hits) painted with green
overlay[gt & avg_pred] = np.array([0, 250, 0], dtype=overlay.dtype)
# Misses painted with red
overlay[gt & ~avg_pred] = np.array([250, 0, 0], dtype=overlay.dtype)
# False alarm painted with yellow
overlay[~gt & avg_pred] = np.array([250, 250, 0], dtype=overlay.dtype)

overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

visualize(image, avg_pred, gt, overlay)
plt.show()