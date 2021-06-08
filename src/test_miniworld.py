import glob
import rasterio
import imagesize

train_labeled_image_paths = sorted(
    glob.glob(f'/scratch_ai4geo/miniworld_tif/christchurch/train/*_x.tif')
)
test_labeled_image_paths = sorted(
    glob.glob(f'/scratch_ai4geo/miniworld_tif/christchurch/test/*_x.tif')
)

test_local = sorted(
    glob.glob(f'/home/pierre/Documents/ONERA/ai4geo/miniworld_tif/austin/train/*_x.tif')
)
# test_local = train_labeled_image_paths+test_labeled_image_paths

for image in test_local:
    width, height = imagesize.get(image)
    if width != 1500 or height!=1500:
        print(image)