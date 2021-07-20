import glob
from argparse import ArgumentParser
from torch_datasets import MultipleImages
import numpy as np

parser = ArgumentParser()
parser.add_argument("--paths", type=str)
args = parser.parse_args()

paths = glob.glob(args.path)
dataset = MultipleImages(
    images_paths=paths,
    crop_size=128
)
out = {
    'min': None,
    'max': None,
    'mean': None,
    'std': None
}
for i in range(100):
    idx = np.random.randint(len(dataset))
    image = dataset[idx]['image']
    m = np.min(image, axis=[1,2])
    M = np.max(image, axis=[1,2])
    mean = np.mean(image, axis=[1,2])
    std = np.std(image, axis=[1,2])
    if out is None:
        out = np.zeros(shape=(image.shape[0], 4))
    for j in range(image.shape[0]):
        out[j][0] = max(out[j][0], min(image[j, :, :])) # computing min

mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)
num_im = 0
print('==> Computing mean and std..')
indices = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
for city_info in Miniworld.city_info_list:
    for i in range(city_info[2]):
        filepath = '/scratch_ai4geo/miniworld/{}/train/{}_x.png'.format(
            city_info[0], i)
        with rasterio.open(filepath) as image:
            im = image.read()
            for i in range(N_CHANNELS):
                mean[i] += im[i,:,:].mean()
                std[i] += im[i,:,:].std()
            num_im += 1
        if i>1000:
            break
mean.div_(num_im*255)
std.div_(num_im*255)
print(mean, std)