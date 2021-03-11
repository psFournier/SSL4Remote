import numpy as np


class MergeLabels:
    def __init__(self, labels):

        self.labels = labels

    def __call__(self, L):

        ret = np.zeros(L.shape, dtype=L.dtype)
        for i, lab in enumerate(self.labels):
            for j in lab:
                ret[L == j] = i

        return ret
