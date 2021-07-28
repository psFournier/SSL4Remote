from dl_toolbox.torch_datasets import MultipleImages
import torch
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

class MiniworldCityDs(MultipleImages):
        
    def __init__(self, city, *args, **kwargs):
        
        super(MiniworldCityDs, self).__init__(*args, **kwargs)
        self.city = city

    def process_label(self, label):

        if self.city in ['chicago', 'vienna', 'tyrol-w', 'austin', 'kitsap', 'christchurch']:
            label = inria_label_formatter(label)

        if self.city in ['toulouse']:
            label = semcity_label_formatter(label)

        if self.city in ['potsdam']:
            label = isprs_label_formatter(label)

        mask = torch.from_numpy(label).contiguous()

        return mask

    def process_image(self, image):

        if self.city in ['toulouse']:
            return np.uint8(image[[4,3,2], :, :]/16)/255
        else:
            return super(MiniworldCityDs, self).process_image(image)

