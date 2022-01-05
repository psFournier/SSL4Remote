from torch_datasets import OneImage
import torch
import numpy as np


class PhrPanDs(OneImage):

    def __init__(self, *args, **kwargs):

        super(PhrPanDs, self).__init__(*args, **kwargs)

    def process_image(self, image):

        return torch.from_numpy(image[[0,1,2], :, :]).contiguous()

    # def process_label(self, label):
    #
    #     labels0 = np.zeros(shape=label.shape[1:], dtype=float)
    #     labels1 = np.zeros(shape=label.shape[1:], dtype=float)
    #     mask = label[0, :, :] == 3
    #     np.putmask(labels0, ~mask, 1.)
    #     np.putmask(labels1, mask, 1.)
    #     label = np.stack([labels0, labels1], axis=0)
    #     label = torch.from_numpy(label).contiguous()
    #
    #     return label

    def process_label(self, label):

        labels = [np.zeros(shape=label.shape[1:], dtype=float) for _ in range(10)]
        masks = [label[0, :, :] == i for i in range(1,10)]
        any_label_mask = np.logical_or.reduce(masks)
        masks = [~any_label_mask] + masks
        for i in range(10):
            np.putmask(labels[i], masks[i], 1.)
        label = np.stack(labels, axis=0)
        label = torch.from_numpy(label).contiguous()

        return label
