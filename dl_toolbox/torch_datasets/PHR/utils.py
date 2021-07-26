import numpy as np
import torch


def binary_labels_to_rgb(labels):

    labels = labels.cpu().numpy()
    colors = np.zeros(shape=(labels.shape[0], labels.shape[1], labels.shape[2], 3), dtype=np.uint8)
    idx = np.array(labels == 1)
    colors[idx] = np.array([255,255,255])
    res = np.transpose(colors, axes=(0, 3, 1, 2))
    return torch.from_numpy(res).float()


def phr_binary_labels(labels):

    labels0 = np.zeros(shape=labels.shape[1:], dtype=float)
    labels1 = np.zeros(shape=labels.shape[1:], dtype=float)
    mask = labels[0, :, :] == 5
    np.putmask(labels0, ~mask, 1.)
    np.putmask(labels1, mask, 1.)
    labels = np.stack([labels0, labels1], axis=0)

    return labels