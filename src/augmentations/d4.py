import torch
import torchvision.transforms.functional as F
import random

class Vflip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label):

        if torch.rand(1).item() < self.p:
            return F.vflip(img), F.vflip(label)

        return img, label

class Hflip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label):

        if torch.rand(1).item() < self.p:
            return F.hflip(img), F.hflip(label)

        return img, label

class Transpose(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label):

        if torch.rand(1).item() < self.p:
            return torch.transpose(img, 2, 3), torch.transpose(label, 2, 3)

        return img, label

class Rotate(torch.nn.Module):

    def __init__(self, p=0.5, angles=(90, 180, 270)):
        super().__init__()
        self.p = p
        self.angles = angles

    def __call__(self, img, label):

        if torch.rand(1).item() < self.p:
            angle = random.choice(self.angles)
            return F.rotate(img, angle), F.rotate(label, angle)

        return img, label

class D4(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = [
            Rotate(angles=(0, 90, 180, 270)),
            Hflip(),
            Vflip(),
            Transpose()
        ]

    def __call__(self, img, label):

        if torch.rand(1).item() < self.p:
            for transform in self.transforms:
                img, label = transform(img, label)

        return img, label
