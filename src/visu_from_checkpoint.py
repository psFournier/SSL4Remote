from pl_modules import SupervisedBaseline
import rasterio as rio
import numpy as np
from rasterio.windows import Window
import matplotlib.pyplot as plt
from torch_datasets import Paris
from utils import MergeLabels
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

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


ckpt_path = '/home/pierre/PycharmProjects/RemoteSensing/outputs/tensorboard/baseline_paris_2021-04-24/checkpoints/epoch=999-step=312999.ckpt'

module = SupervisedBaseline.load_from_checkpoint(ckpt_path)
crop_size = 350

module.eval()

image_path = '/home/pierre/Documents/ONERA/ai4geo/small_paris/paris/test/100_x.png'
label_path = '/home/pierre/Documents/ONERA/ai4geo/small_paris/paris/test/100_y.png'

with rio.open(image_path) as image_file:

    cols = image_file.width
    rows = image_file.height
    cx = np.random.randint(0, cols - crop_size - 1)
    cy = np.random.randint(0, rows - crop_size - 1)
    window = rio.windows.Window(cx, cy, crop_size, crop_size)
    image = image_file.read(window=window,
                            out_dtype=np.uint8).transpose(1, 2, 0)

with rio.open(label_path) as label_file:
    label = label_file.read(window=window, out_dtype=np.uint8).transpose(1, 2, 0)

gt = Paris.colors_to_labels(label)

label_merger = MergeLabels([[0], [1]])
gt = label_merger(gt).astype(bool)

augment = A.Compose([
    A.Normalize(),
    ToTensorV2(transpose_mask=False)
])
augmented = augment(image=image)['image'].reshape(1,3,crop_size, crop_size)
pred = module.network(augmented)
pred = np.argmax(pred.detach().numpy().squeeze(), axis=0).astype(bool)
#
overlay = image.copy()
# Correct predictions (Hits) painted with green
overlay[gt & pred] = np.array([0, 250, 0], dtype=overlay.dtype)
# Misses painted with red
overlay[gt & ~pred] = np.array([250, 0, 0], dtype=overlay.dtype)
# False alarm painted with yellow
overlay[~gt & pred] = np.array([250, 250, 0], dtype=overlay.dtype)

overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

visualize(image, pred, gt, overlay)
plt.show()