import torch
import torchvision.transforms.functional as F
import random
from dl_toolbox.augmentations import Compose, NoOp

class Vflip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = F.vflip(img)
            if label is not None: label = F.vflip(label)
            return img, label

        return img, label

class Hflip:

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = F.hflip(img)
            if label is not None: label = F.hflip(label)
            return img, label

        return img, label

class Transpose1(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = torch.transpose(img, -2, -1)
            if label is not None: label = torch.transpose(label, -2, -1)
            return img, label

        return img, label

class Transpose2(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = img.flip(-1)
            img = img.flip(-2)
            img = torch.transpose(img, -2, -1)
            if label is not None:
                label = label.flip(-1)
                label = label.flip(-2)
                label = torch.transpose(label, -2, -1)
            return img, label

        return img, label

class Rot90:

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = torch.transpose(F.hflip(img), -2, -1)
            if label is not None:
                label = torch.transpose(F.hflip(label), -2, -1)
            return img, label

        return img, label


class Rot180:

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = F.vflip(F.hflip(img))
            if label is not None:
                label = F.vflip(F.hflip(label))
            return img, label

        return img, label


class Rot270:

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = torch.transpose(F.vflip(img), -2, -1)
            if label is not None:
                label = torch.transpose(F.vflip(label), -2, -1)
            return img, label

        return img, label

# class Rotate(torch.nn.Module):
#
#     def __init__(self, p=0.5, angles=(90, 180, 270)):
#         super().__init__()
#         self.p = p
#         self.angles = angles
#
#     def __call__(self, img, label=None):
#
#         if torch.rand(1).item() < self.p:
#             angle = random.choice(self.angles)
#             img = F.rotate(img, angle)
#             if label is not None: label = F.rotate(label, angle)
#             return img, label
#
#         return img, label

class D4(torch.nn.Module):

    def __init__(self, p=1):
        super().__init__()
        self.transforms = [
            NoOp(),
            Hflip(p=1),
            Vflip(p=1),
            Transpose1(p=1),
            Transpose2(p=1),
            Rot90(p=1),
            Rot180(p=1),
            Rot270(p=1)
        ]

    def __call__(self, img, label):

        t = random.choice(self.transforms)

        return t(img, label)
