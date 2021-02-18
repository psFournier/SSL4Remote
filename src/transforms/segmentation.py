import numpy as np

class Albu_image_transfo:

    def __init__(self, transform):

        self.transform = transform

    def __call__(self, image):

        return self.transform(image=image)['image']

class Albu_image_label_transfo:

    def __init__(self, transform):

        self.transform = transform

    def __call__(self, image, label):

        transformed = self.transform(image=image,
                                     mask=label)

        return transformed['image'], transformed['mask']

class Merge_labels:

    def __init__(self, labels):

        self.labels = labels

    def __call__(self, L):

        ret = np.zeros(L.shape, dtype=L.dtype)
        for i, lab in enumerate(self.labels):
            for j in lab:
                ret[L == j] = i

        return ret