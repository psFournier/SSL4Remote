import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

list_files = [os.path.join('/d/pfournie/Documents/ai4geo/data/SemcityTLS_DL/', path) for path in [
    'train/TLS_GT_03.tif',
    'train/TLS_GT_08.tif',
    'val/TLS_GT_07.tif',
    'val/TLS_GT_04.tif'
]]

sources = [rasterio.open(file) for file in list_files]

mosaic, out_trans = merge(sources)
# show(mosaic, cmap='terrain')

out_meta = sources[0].meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 # "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                 })

out_path = '/d/pfournie/Documents/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif'
with rasterio.open(out_path, "w", **out_meta) as dest:
    dest.write(mosaic)