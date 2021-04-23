from torch_datasets import AirsLabeled, IsprsVLabeled
import matplotlib.pyplot as plt
import albumentations as A
from torch.utils.data import DataLoader

ds = AirsLabeled(
    data_path='/home/pierre/Documents/ONERA/ai4geo/small_airs',
    idxs=list(range(9)),
    crop=512,
    augmentations=A.NoOp()
)

dl = DataLoader(
    ds,
    batch_size=1,
)

image, mask = next(iter(dl))
image = image[0,...].numpy()
mask = mask[0,...].numpy()
plt.imshow(image)
plt.show()
