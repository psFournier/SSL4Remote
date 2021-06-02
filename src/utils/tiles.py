from itertools import product
from rasterio.windows import Window
import numpy as np

def get_tiles(ds, nols, nrows, width=256, height=256, col_step=128,
              row_step=128):

    assert col_step <= width
    assert row_step <= height
    assert nols <= ds.meta['width']
    assert nrows <= ds.meta['height']

    max_col_offset = int(np.ceil((nols-width)/col_step))
    # Remove all offsets such that offset+width > nols and add one offset to
    # reach nols
    col_offsets = list(range(0, nols, col_step))[:max_col_offset+1]
    col_offsets[max_col_offset] = nols - width

    max_row_offset = int(np.ceil((nrows-height)/row_step))
    # Remove all offsets such that offset+width > nols and add one offset to
    # reach nols
    row_offsets = list(range(0, nrows, row_step))[:max_row_offset+1]
    row_offsets[max_row_offset] = nrows - height

    offsets = product(col_offsets, row_offsets)
    big_window = Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = Window(col_off=col_off, row_off=row_off, width=width,
                        height=height).intersection(big_window)
        yield window
