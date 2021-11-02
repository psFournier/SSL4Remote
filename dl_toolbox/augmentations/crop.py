from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F


class RandomCrop2(RandomCrop):

    def __init__(self, size):
        super(RandomCrop2, self).__init__(size, padding=None, pad_if_needed=False)

    def forward(self, img, label=None):

        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        if label is not None: label = F.crop(label, i, j, h, w)

        return img, label