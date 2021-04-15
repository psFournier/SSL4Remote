import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A


aug = A.ShiftScaleRotate()
class RandomDataset(Dataset):

    def __getitem__(self, index):
        image = np.random.randint(0,255, size=(128,128,3), dtype=np.uint8)
        label = np.random.randint(0, 255, size=(128,128,3), dtype=np.uint8)
        res = aug(image=image, mask=label)
        return res['image'], res['mask']

    def __len__(self):
        return 2

ds = DataLoader(
    RandomDataset(),
    batch_size=2,
)

for epoch in range(20):
    print("epoch {}".format(epoch))
    for batch in ds:
        image, label = batch
        print(label.dtype)