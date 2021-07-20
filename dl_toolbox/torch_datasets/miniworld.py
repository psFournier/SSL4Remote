import numpy as np

def inria_label_formatter(labels):
    '''
    Creates one-hot encoded labels for binary classification.
    :param labels_color:
    :return:
    '''

    labels0 = np.zeros(shape=labels.shape[1:], dtype=float)
    labels1 = np.zeros(shape=labels.shape[1:], dtype=float)
    mask = np.any(labels != [0], axis=0)
    np.putmask(labels0, ~mask, 1.)
    np.putmask(labels1, mask, 1.)
    labels = np.stack([labels0, labels1], axis=0)

    return labels

def isprs_label_formatter(labels):

    labels0 = np.zeros(shape=labels.shape[1:], dtype=float)
    labels1 = np.zeros(shape=labels.shape[1:], dtype=float)
    mask = np.logical_and(
        labels[0, :, :] == 0,
        labels[1, :, :] == 0,
        labels[2, :, :] == 255,
    )
    np.putmask(labels0, ~mask, 1.)
    np.putmask(labels1, mask, 1.)
    labels = np.stack([labels0, labels1], axis=0)

    return labels

def semcity_label_formatter(labels):

    labels0 = np.zeros(shape=labels.shape[1:], dtype=float)
    labels1 = np.zeros(shape=labels.shape[1:], dtype=float)
    mask = np.logical_and(
        labels[0, :, :] == 238,
        labels[1, :, :] == 118,
        labels[2, :, :] == 33,
    )
    np.putmask(labels0, ~mask, 1.)
    np.putmask(labels1, mask, 1.)
    labels = np.stack([labels0, labels1], axis=0)

    return labels

def miniworld_label_formatter(labels, city):

    if city in ['chicago', 'vienna', 'tyrol-w', 'austin', 'kitsap', 'christchurch']:
        return inria_label_formatter(labels)

    if city in ['toulouse']:
        return semcity_label_formatter(labels)

    if city in ['potsdam']:
        return isprs_label_formatter(labels)