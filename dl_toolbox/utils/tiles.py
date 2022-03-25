from itertools import product
from rasterio.windows import Window
import numpy as np

def get_tiles(nols, nrows, size, size2=None, step=None, step2=None, col_offset=0, row_offset=0):
    
    if step is None: step = size
    if size2 is None: size2 = size
    if step2 is None: step2 = step

    max_col_offset = int(np.ceil((nols-size)/step))
    # Remove all offsets such that offset+size > nols and add one offset to
    # reach nols
    col_offsets = list(range(col_offset, col_offset + nols, step))[:max_col_offset+1]
    col_offsets[max_col_offset] = col_offset + nols - size

    max_row_offset = int(np.ceil((nrows-size2)/step2))
    # Remove all offsets such that offset+size > nols and add one offset to
    # reach nols
    row_offsets = list(range(row_offset, row_offset + nrows, step2))[:max_row_offset+1]
    row_offsets[max_row_offset] = row_offset + nrows - size2

    offsets = product(col_offsets, row_offsets)
    big_window = Window(col_off=col_offset, row_off=row_offset, width=nols, height=nrows)
    for col_off, row_off in offsets:
        window = Window(col_off=col_off, row_off=row_off, width=size,
                        height=size2).intersection(big_window)
        yield window

def main():

    for tile in get_tiles(1000,1500,412, size2=397, step=400, col_offset=10):
        print(tile)


if __name__ == "__main__":

    main()
