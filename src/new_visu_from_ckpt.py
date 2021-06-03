from pl_modules import SupervisedBaseline, MeanTeacher
import rasterio as rio
import matplotlib.pyplot as plt
from torch_datasets import *
import cv2
import torch
from utils import get_tiles
from argparse import ArgumentParser
import torchvision.transforms.functional as F
from augmentations import *

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

def infer(module,
          img_file,
          height,
          width,
          tile_height,
          tile_width,
          col_step,
          row_step,
          batch_size):

    windows = []
    tiles = []
    for window in get_tiles(width=tile_width,
                            height=tile_height,
                            col_step=col_step,
                            row_step=row_step,
                            nols=width,
                            nrows=height):

        tile = img_file.read(window=window, out_dtype=np.float32) / 255
        tiles.append(torch.as_tensor(tile))
        windows.append(window)

    nb_tiles = len(tiles)
    l = list(range(0, nb_tiles, batch_size))+[nb_tiles]

    preds = []
    pred_windows = []
    for i, j in zip(l[:-1], l[1:]):

        batch = torch.stack(tiles[i:j])

        for angle in [0,90,270]:
            for ph in [0, 1]:
                for pv in [0, 1]:
                    for pt in [0, 1]:
                        t = Compose([
                            Rotate(p=1, angles=(angle,)),
                            Hflip(p=ph),
                            Vflip(p=pv),
                            Transpose(p=pt)
                        ])
                        aug_batch = t(batch)[0]
                        aug_pred = module.network(aug_batch).detach()
                        anti_t = Compose([
                            Transpose(p=pt),
                            Vflip(p=pv),
                            Hflip(p=ph),
                            Rotate(p=1, angles=(-angle,))
                        ])
                        pred = anti_t(aug_pred)[0].numpy()
                        preds += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]
                        pred_windows += windows[i:j]

    stitched_pred = np.zeros(shape=(2, height, width))
    nb_tiles = np.zeros(shape=(height, width))
    for pred, window in zip(preds, pred_windows):
        stitched_pred[:, window.row_off:window.row_off+window.width,
        window.col_off:window.col_off+window.height] += pred
        nb_tiles[window.row_off:window.row_off+window.width,
        window.col_off:window.col_off+window.height] += 1
    avg_pred = np.argmax(stitched_pred, axis=0)

    return avg_pred


width = 500
height = 500
with rio.open(image_path) as image_file:

    image = image_file.read(window=Window(col_off=0,
                                          row_off=0,
                                          width=width,
                                          height=height),
                            out_dtype=np.float32) / 255

    avg_pred = infer(module,
                     image_file,
                     height=height,
                     width=width,
                     tile_height=256,
                     tile_width=256,
                     col_step=128,
                     row_step=128,
                     batch_size=16)

    bool_avg_pred = avg_pred.astype(bool)



with rio.open(label_path) as label_file:
    label = label_file.read(window=Window(col_off=0,
                                          row_off=0,
                                          width=width,
                                          height=height),
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

