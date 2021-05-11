import torch
import torchvision.transforms.functional as F

class Sharpness(torch.nn.Module):

    def __init__(self, bounds=(0.8, 1.2), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label=None):

        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_sharpness(img, factor), label
        return img, label
