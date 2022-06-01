import os
import rasterio
from rasterio.merge import merge
from rasterio.plot import show

list_files = [os.path.join('/d/pfournie/ai4geo/data/SemcityTLS_DL/', path) for path in [
    'train/TLS_BDSD_M_03.tif',
    'train/TLS_BDSD_M_08.tif',
    'val/TLS_BDSD_M_07.tif',
    'val/TLS_BDSD_M_04.tif',
    'test/TLS_BDSD_M_01.tif',
    'test/TLS_BDSD_M_02.tif',
    'test/TLS_BDSD_M_05.tif',
    'test/TLS_BDSD_M_06.tif',
    'test/TLS_BDSD_M_09.tif',
    'test/TLS_BDSD_M_10.tif',
    'test/TLS_BDSD_M_11.tif',
    'test/TLS_BDSD_M_12.tif',
    'test/TLS_BDSD_M_13.tif',
    'test/TLS_BDSD_M_14.tif',
    'test/TLS_BDSD_M_15.tif',
    'test/TLS_BDSD_M_16.tif',
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

out_path = '/d/pfournie/ai4geo/data/SemcityTLS_DL/BDSD_M.tif'
with rasterio.open(out_path, "w", **out_meta) as dest:
    dest.write(mosaic)
