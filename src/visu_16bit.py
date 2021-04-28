import random
import rasterio
import numpy as np
import cv2
from matplotlib import pyplot as plt

import albumentations as A

def visualize(image):
    # Divide all values by 65535 so we can display the image using matplotlib
    image = image / 65535
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

raster = rasterio.open("/home/pierre/Documents/ONERA/ai4geo/PHR_Paris/window.tif")
image = raster.read(out_dtype=np.uint16).transpose(1,2,0) / 255
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(image[:, :, 1:])
plt.show()