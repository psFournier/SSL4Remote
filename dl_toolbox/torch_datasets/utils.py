import numpy as np
import rasterio


def minmax(image, m, M):
    
    return np.clip((image - np.reshape(m, (-1, 1, 1))) / np.reshape(M - m, (-1, 1, 1)), 0, 1)

def read_window_from_big_raster(window, path, raster_path):
    with rasterio.open(path) as image_file:
        with rasterio.open(raster_path) as raster_file:
            minx, miny, maxx, maxy = rasterio.windows.bounds(
                window, 
                transform=image_file.transform
            )
            window_in_original_raster = rasterio.windows.from_bounds(
                minx, miny, maxx, maxy, 
                transform=raster_file.transform
            )
            image = raster_file.read(
                window=window_in_original_raster, 
                out_dtype=np.float32
            )
    return image

def read_window_basic(window, path):
    with rasterio.open(path) as image_file:
        image = image_file.read(window=window, out_dtype=np.float32)
    return image

