import torch
import torchvision.transforms.functional as F

class Gamma(torch.nn.Module):

    def __init__(self, bounds=(0.8, 1.2), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_gamma(img, factor), label

        return img, label


class Saturation(torch.nn.Module):

    def __init__(self, bounds=(0.8, 1.2), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_saturation(img, factor), label
        return img, label


class Hue(torch.nn.Module):

    def __init__(self, bounds=(-0.2, 0.2), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_hue(img, factor), label
        return img, label


class Brightness(torch.nn.Module):

    def __init__(self, bounds=(0.8, 1.2), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_brightness(img, factor), label
        return img, label


class Contrast(torch.nn.Module):

    def __init__(self, bounds=(0.8, 1.2), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_contrast(img, factor), label
        return img, label