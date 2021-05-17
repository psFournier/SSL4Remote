import torch
import torchvision.transforms.functional as F
import random

class D4(torch.nn.Module):

    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def __call__(self, img, label):

        angle = random.choice([0, 90, 270])
        img = F.rotate(img, angle)
        label = F.rotate(label, angle)

        if torch.rand(1).item() < self.p:
            img = F.hflip(img)
            label = F.hflip(label)

        if torch.rand(1).item() < self.p:
            img = F.vflip(img)
            label = F.vflip(label)

        if torch.rand(1).item() < self.p:
            img = torch.transpose(img, 2, 3)
            label = torch.transpose(label, 2, 3)

        return img, label
