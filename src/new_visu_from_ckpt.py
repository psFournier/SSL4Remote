from pl_modules import SupervisedBaseline, MeanTeacher
import rasterio as rio
import matplotlib.pyplot as plt
from torch_datasets import *
import cv2
import torch
from utils import get_tiles
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default='/home/pierre/PycharmProjects/RemoteSensing/outputs')
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--image_dir", type=str, default='/home/pierre/Documents/ONERA/ai4geo/miniworld_tif')
parser.add_argument("--image_path", type=str)
parser.add_argument("--label_path", type=str)
args = parser.parse_args()
args_dict = vars(args)

ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_path)
ckpt = torch.load(ckpt_path)
module = SupervisedBaseline()
module.load_state_dict(ckpt['state_dict'])
module.eval()

image_path = os.path.join(args.image_dir, args.image_path)
label_path = os.path.join(args.image_dir, args.label_path)

windows = []
tiles = []
full_height = 257
full_width = 257
with rio.open(image_path) as image_file:

    image = image_file.read(window=Window(col_off=0,
                                          row_off=0,
                                          width=full_width,
                                          height=full_height),
                            out_dtype=np.float32) / 255

    for window in get_tiles(width=128, height=128, col_step=128,
                            row_step=128, nols=full_width, nrows=full_height):

        tile = image_file.read(window=window, out_dtype=np.float32) / 255
        tiles.append(torch.as_tensor(tile))
        windows.append(window)

preds = []
i=0
while i+16 < len(tiles):
    batch = torch.stack(tiles[i:i+16], dim=0)
    pred = module.network(batch)
    pred = np.argmax(pred.detach().numpy().squeeze(), axis=1).astype(bool)
    preds += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0],
                                                      axis=0)]
    i += 16
batch = torch.stack(tiles[i:], dim=0)
pred = module.network(batch)
pred = np.argmax(pred.detach().numpy().squeeze(), axis=1).astype(bool)
preds += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0],
                                                  axis=0)]

stitched_pred = np.zeros(shape=(full_height,full_width))
nb_tiles = np.zeros(shape=(full_height, full_width))
for pred, window in zip(preds, windows):
    stitched_pred[window.row_off:window.row_off+window.width,
    window.col_off:window.col_off+window.height] += pred
    nb_tiles[window.row_off:window.row_off+window.width,
    window.col_off:window.col_off+window.height] += 1
avg_pred = np.rint(np.divide(stitched_pred, nb_tiles))
bool_avg_pred = avg_pred.astype(bool)
# bool_uncertain = (avg_pred != avg_pred.round())


with rio.open(label_path) as label_file:
    label = label_file.read(window=Window(col_off=0,
                                          row_off=0,
                                          width=full_width,
                                          height=full_height),
                            out_dtype=np.uint8)

gt = AustinLabeled.colors_to_labels(label)
gt = np.argmax(gt, axis=0).astype(bool)

overlay = image.copy().transpose(1,2,0)
image = image.transpose(1,2,0)
# Correct predictions (Hits) painted with green
overlay[gt & bool_avg_pred] = np.array([0, 250, 0], dtype=overlay.dtype)
# Misses painted with red
overlay[gt & ~bool_avg_pred] = np.array([250, 0, 0], dtype=overlay.dtype)
# False alarm painted with blue
overlay[~gt & bool_avg_pred] = np.array([0, 0, 250], dtype=overlay.dtype)

overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

# uncertainty = image.copy()
# uncertainty[bool_uncertain] = np.array([250, 0, 0], dtype=overlay.dtype)
# uncertainty = cv2.addWeighted(image, 0.5, uncertainty, 0.5, 0)

plt.switch_backend("TkAgg")

f, ax = plt.subplots(2, 2, figsize=(16, 16))

ax[0, 0].imshow(image)
ax[0, 0].set_title('Original image', fontsize=16)

ax[1, 0].imshow(gt)
ax[1, 0].set_title('Ground truth', fontsize=16)

ax[0, 1].imshow(avg_pred)
ax[0, 1].set_title('Model predictions', fontsize=16)

ax[1, 1].imshow(overlay)
ax[1, 1].set_title('Prediction quality', fontsize=16)

plt.tight_layout()
plt.show()

