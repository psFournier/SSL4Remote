import os
from itertools import product
import rasterio as rio
from rasterio import windows

in_path = '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN/top'
input_filename = 'top_mosaic_09cm_area10.tif'

out_path = '/home/pierre/Documents/ONERA/ai4geo/ISPRS_VAIHINGEN/top/tiles'
output_filename = '{}-tile_{}-{}.tif'

def get_tiles(ds, width=256, height=256, col_step=128, row_step=128):
    assert col_step <= width
    assert row_step <= height
    nols, nrows = ds.meta['width'], ds.meta['height']

    max_col_offset = int((nols-width)/col_step)+1
    # Remove all offsets such that offset+width > nols and add one offset to
    # reach nols
    col_offsets = list(range(0, nols, col_step))[:max_col_offset+1]
    col_offsets[max_col_offset] = nols - width

    max_row_offset = int((nrows - height) / row_step) + 1
    # Remove all offsets such that offset+width > nols and add one offset to
    # reach nols
    row_offsets = list(range(0, nrows, row_step))[:max_row_offset+1]
    row_offsets[max_row_offset] = nrows - height

    offsets = product(col_offsets, row_offsets)
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width,
                               height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


with rio.open(os.path.join(in_path, input_filename)) as inds:
    tile_width, tile_height = 256, 256

    meta = inds.meta.copy()

    for window, transform in get_tiles(inds):
        print(window)
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        outpath = os.path.join(out_path,output_filename.format(
            os.path.splitext(input_filename)[0],
            int(window.col_off),
            int(window.row_off)))
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(inds.read(window=window))
