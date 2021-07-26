import numpy as np
import torch

class NoOp():

    def __init__(self):

        pass

    def __call__(self, img, label=None):

        return img, label

class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label=None):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class OneOf:

    def __init__(self, transforms, transforms_ps):
        self.transforms = transforms
        s = sum(transforms_ps)
        self.transforms_ps = [p / s for p in transforms_ps]

    def __call__(self, img, label=None):
        t = np.random.choice(self.transforms, p=self.transforms_ps)
        img, label = t(img, label)
        return img, label


def rand_bbox(size, lam):

    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

