import rasterio
from rasterio.windows import Window
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

raster = rasterio.open(args.src)
out_profile = raster.profile
w = raster.read(window=Window(0, 0, 1000, 1000))
_, height, width = w.shape
out_profile.update(
    height=height,
    width=width,
)
with rasterio.open(args.dst, 'w', **out_profile) as dst:
    dst.write(w)
