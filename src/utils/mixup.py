import numpy as np



class Cutmix():

    def __init__(self, alpha):

        self.alpha = alpha

    @staticmethod
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

    def __call__(self, batch):

        lam = np.random.beta(self.alpha, self.alpha)
        idx = np.random.permutation(len(batch))
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
        perm = [batch[i] for i in idx]
        mixed_batch = [
            (lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2) for (x1, y1), (x2, y2) in zip(batch, perm)
        ]

        return mixed_batch

