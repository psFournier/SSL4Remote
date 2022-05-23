import numpy as np
import rasterio


def minmax(image, m, M):
    
    return np.clip((image - np.reshape(m, (-1, 1, 1))) / np.reshape(M - m, (-1, 1, 1)), 0, 1)

def read_window_from_big_raster(window, image_file, big_raster):
    minx, miny, maxx, maxy = rasterio.windows.bounds(window, transform=image_file.transform)
    window_in_original_raster = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=big_raster.transform)
    image = big_raster.read(window=window_in_original_raster, out_dtype=np.float32)
    return image

