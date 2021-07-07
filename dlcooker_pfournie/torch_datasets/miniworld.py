import numpy as np

def colors_to_labels(colors):

    '''
    Creates one-hot encoded labels for binary classification.
    :param labels_color:
    :return:
    '''

    labels0 = np.zeros(shape=colors.shape[1:], dtype=float)
    labels1 = np.zeros(shape=colors.shape[1:], dtype=float)
    mask = np.any(colors != [0], axis=0)
    np.putmask(labels0, ~mask, 1.)
    np.putmask(labels1, mask, 1.)
    labels = np.stack([labels0, labels1], axis=0)

    return labels