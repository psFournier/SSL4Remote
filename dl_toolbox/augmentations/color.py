import torch
import torchvision.transforms.functional as F
import dl_toolbox.augmentations as aug

class Gamma(torch.nn.Module):

    def __init__(self, bound=0.5, p=2.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def apply(self, img, label=None, factor=1.):

        return F.adjust_gamma(img, factor), label

    def forward(self, img, label=None):

        if torch.rand(1).item() < self.p:
            factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
            return self.apply(img, label, factor)

        return img, label


class Saturation(torch.nn.Module):

    def __init__(self, bound=0.5, p=0.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_saturation(img, factor), label
        return img, label

class Brightness(torch.nn.Module):

    def __init__(self, bound=0.2, p=0.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_brightness(img, factor), label
        return img, label


class Contrast(torch.nn.Module):

    def __init__(self, bound=0.4, p=0.5):
        super().__init__()
        self.bounds = (1-bound, 1+bound)
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_contrast(img, factor), label
        return img, label

class Color():

    def __init__(self, bound):
        self.bound = bound
        self.color_aug = aug.Compose(
            [
                Saturation(p=1, bound=bound),
                Contrast(p=1, bound=bound),
                Gamma(p=1, bound=bound),
                Brightness(p=1, bound=bound)
            ]
        )

    def __call__(self, image, label):
        return self.color_aug(image, label)
