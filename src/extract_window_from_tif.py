import rasterio
from rasterio.windows import Window

raster = rasterio.open("/work/OT/ai4geo/DATA/REF/COS_4/PARIS/20180326/IMG_PHR1A_PMS_201803261058346_SEN_3625348101-006_R1C1_PROJECTED_g.tif")
out_profile = raster.profile
w = raster.read(window=Window(0, 0, 1000, 1000))
_, height, width = w.shape
out_profile.update(
    height=height,
    width=width,
)
with rasterio.open("/home/eh/fournip/SemiSupervised/SSL4Remote/paris.tif", 'w', **out_profile) as dst:
    dst.write(w)
