import numpy as np
import rasterio as rio
from torch_datasets import IsprsVaihingen
from transforms import MergeLabels
import matplotlib.pyplot as plt
import cv2
from rasterio.windows import Window
import albumentations as A

plt.switch_backend("TkAgg")


x_path = '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN/top/top_mosaic_09cm_area1.tif'
gt_path = '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN' \
          '/gts_for_participants' \
         '/top_mosaic_09cm_area1.tif'

size = 1024
window = Window(0, 0, size, size)

x_src = rio.open(x_path)
ncols = x_src.width
nrows = x_src.height
top = x_src.read(window=window).transpose(1, 2, 0)

gt_src = rio.open(gt_path)
gt = gt_src.read(window=window, out_dtype=np.float32).transpose(1, 2, 0)
gt = IsprsVaihingen.colors_to_labels(gt)

label_merger = MergeLabels([[0], [1]])
gt = label_merger(gt).astype(bool)

pred = np.zeros(shape=(size, size), dtype=np.float32)
pred[10:50:,:] = 1.
pred = pred.astype(bool)
#
overlay = top.copy()
# Correct predictions (Hits) painted with green
overlay[gt & pred] = np.array([0, 250, 0], dtype=overlay.dtype)
# Misses painted with red
overlay[gt & ~pred] = np.array([0, 0, 250], dtype=overlay.dtype)
# False alarm painted with yellow
overlay[~gt & pred] = np.array([250, 250, 0], dtype=overlay.dtype)

overlay = cv2.addWeighted(top, 0.7, overlay, 0.3, 0)

aug = A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1)
image_aug = top.copy()
image_aug = aug(image=image_aug)['image']

f, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].imshow(top)
ax[0].set_title('Original image')

ax[1].imshow(image_aug)
ax[1].set_title('Augmented image')

f.tight_layout()

plt.show()




# checkpoint_path = '/home/pierre/PycharmProjects/RemoteSensing/outputs/tensorboard/MT_isprs_v_2021-04-01/checkpoints/epoch=0-step=0.ckpt'
# model = MeanTeacher.load_from_checkpoint(checkpoint_path)
# model.eval()
# y_hat = model()