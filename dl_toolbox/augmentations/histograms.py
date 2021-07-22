import torch
import torchvision.transforms.functional as F

class HistEq:

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, label):

        if torch.rand(1).item() < self.p:
            return F.equalize(img), label

        return img, label