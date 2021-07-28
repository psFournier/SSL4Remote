from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F


class RandomCrop2(RandomCrop):

    def __init__(self, size):
        super(RandomCrop2).__init__(size, padding=None, pad_if_needed=False)

    def forward(self, img, label=None):

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)