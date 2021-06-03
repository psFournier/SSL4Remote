import torch
import torchvision.transforms.functional as F
import random
from augmentations import Compose

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

class Hflip(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = F.hflip(img)
            if label is not None: label = F.hflip(label)
            return img, label

        return img, label

class Transpose(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            img = torch.transpose(img, 2, 3)
            if label is not None: label = torch.transpose(label, 2, 3)
            return img, label

        return img, label

class Rotate(torch.nn.Module):

    def __init__(self, p=0.5, angles=(90, 180, 270)):
        super().__init__()
        self.p = p
        self.angles = angles

    def __call__(self, img, label=None):

        if torch.rand(1).item() < self.p:
            angle = random.choice(self.angles)
            img = F.rotate(img, angle)
            if label is not None: label = F.rotate(label, angle)
            return img, label

        return img, label

class D4(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.transforms = []
        for angle in [0,90,270]:
            for ph in [0, 1]:
                for pv in [0, 1]:
                    for pt in [0, 1]:
                        self.transforms.append(
                            Compose([
                                Rotate(p=1, angles=(angle,)),
                                Hflip(p=ph),
                                Vflip(p=pv),
                                Transpose(p=pt)
                            ])
                        )

    def __call__(self, img, label):

        t = random.choice(self.transforms)

        return t(img, label)
