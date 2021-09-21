import rasterio
import numpy as np

src = rasterio.open('/d/pfournie/Documents/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif')
img = src.read(out_dtype=np.uint16)
num_channels = img.shape[0]

for i in range(num_channels):
    m, M = np.percentile(img[i, :, :], [1, 99])
    print(m, M)