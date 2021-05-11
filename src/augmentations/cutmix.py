import numpy as np
import torch
from augmentations import rand_bbox

class Cutmix():

    def __init__(self, alpha=0.4):

        self.alpha = alpha

    def __call__(self, batch):

        lam = np.random.beta(self.alpha, self.alpha)
        inputs, targets = batch
        batchsize = inputs.size()[0]
        idx = torch.randperm(batchsize)
        # Use a more generic mask rather than bboxes ?
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        cutmix_inputs, cutmix_targets = inputs, targets
        cutmix_inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[idx, :, bbx1:bbx2, bby1:bby2]
        cutmix_targets[:, :, bbx1:bbx2, bby1:bby2] = targets[idx, :, bbx1:bbx2, bby1:bby2]
        all_inputs = torch.vstack([inputs, cutmix_inputs])
        all_targets = torch.vstack([targets, cutmix_targets])
        idx = np.random.choice(2*batchsize, size=batchsize, replace=False)
        batch = (all_inputs[idx, :], all_targets[idx, :])

        return batch

