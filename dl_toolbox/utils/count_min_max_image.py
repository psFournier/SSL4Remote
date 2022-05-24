import rasterio
import numpy as np

src = rasterio.open('/work/OT/ai4geo/DATA/DATASETS/DIGITANIE/Paris/emprise_ORTHO_cropped.tif')
img = src.read(out_dtype=np.float32)
num_channels = img.shape[0]

for i in range(num_channels):
    m, M = np.percentile(img[i, :, :], [0, 100])
    print(m, M)
