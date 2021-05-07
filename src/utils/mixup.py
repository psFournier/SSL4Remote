import numpy as np

class Mixup():

    def __init__(self, alpha):

        self.alpha = alpha

    def __call__(self, batch):

        lam = np.random.beta(self.alpha, self.alpha)
        idx = np.random.permutation(len(batch))
        perm = [batch[i] for i in idx]
        mixed_batch = [
            (lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2) for (x1, y1), (x2, y2) in zip(batch, perm)
        ]

        return mixed_batch


