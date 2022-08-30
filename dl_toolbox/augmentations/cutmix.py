import numpy as np
import torch
from .utils import rand_bbox

class Cutmix():

    def __init__(self, alpha=0.4):

        self.alpha = alpha

    def __call__(self, input_batch, target_batch):

        lam = np.random.beta(self.alpha, self.alpha)
        batchsize = input_batch.size()[0]
        idx = torch.tensor(np.roll(np.array(range(batchsize)), 1))
        # Use a more generic mask rather than bboxes ?
        bbx1, bby1, bbx2, bby2 = rand_bbox(input_batch.size(), lam)
        cutmix_inputs, cutmix_targets = input_batch, target_batch
        cutmix_inputs[:, ..., bbx1:bbx2, bby1:bby2] = input_batch[idx, ..., bbx1:bbx2, bby1:bby2]
        cutmix_targets[:, ..., bbx1:bbx2, bby1:bby2] = target_batch[idx, ..., bbx1:bbx2, bby1:bby2]

        all_inputs = torch.vstack([input_batch, cutmix_inputs])
        all_targets = torch.vstack([target_batch, cutmix_targets])
        idx = np.random.choice(2*batchsize, size=batchsize, replace=False)
        batch = (all_inputs[idx, :], all_targets[idx, :])

        # batch = (cutmix_inputs, cutmix_targets)

        return batch

class Cutmix2():

    def __init__(self, alpha=0.4):

        self.alpha = alpha

    def __call__(self, input_batch, target_batch):

        lam = np.random.beta(self.alpha, self.alpha)
        batchsize = input_batch.size()[0]
        idx = torch.tensor(np.roll(np.array(range(batchsize)), 1))
        # Use a more generic mask rather than bboxes ?
        bbx1, bby1, bbx2, bby2 = rand_bbox(input_batch.size(), lam)
        cutmix_inputs, cutmix_targets = input_batch, target_batch
        cutmix_inputs[:, ..., bbx1:bbx2, bby1:bby2] = input_batch[idx, ..., bbx1:bbx2, bby1:bby2]
        cutmix_targets[:, ..., bbx1:bbx2, bby1:bby2] = target_batch[idx, ..., bbx1:bbx2, bby1:bby2]

        batch = (cutmix_inputs, cutmix_targets)

        return batch
