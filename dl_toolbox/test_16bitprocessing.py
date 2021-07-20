import rasterio
import numpy as np
import matplotlib.pyplot as plt

image = rasterio.open('/d/pfournie/Documents/ai4geo/data/miniworld_tif/toulouse/test/1_x.tif').read(out_dtype=np.float32)
rgb = image[[4,3,2], :600, :600]
out = np.zeros_like(rgb).astype(np.float32)
for i in range(3):
    c = rgb[i, :, :].min()
    d = rgb[i, :, :].max()

    t = (rgb[i, :, :] - c) / (d - c)
    out[i, :, :] = t
out = out*(2**8-1)
out = rgb / 255
plt.imshow(out.astype(np.uint8).transpose(1,2,0))
plt.show()