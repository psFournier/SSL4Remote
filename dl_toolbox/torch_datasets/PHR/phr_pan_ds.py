from dl_toolbox.torch_datasets import OneImage
import torch
import numpy as np


class PhrPanDs(OneImage):

    def __init__(self, *args, **kwargs):

        super(PhrPanDs, self).__init__(*args, **kwargs)

    def process_image(self, image):

        return torch.from_numpy(image[[0,1,2], :, :]).contiguous()

    def process_label(self, label):

        labels0 = np.zeros(shape=label.shape[1:], dtype=float)
        labels1 = np.zeros(shape=label.shape[1:], dtype=float)
        mask = label[0, :, :] == 3
        np.putmask(labels0, ~mask, 1.)
        np.putmask(labels1, mask, 1.)
        label = np.stack([labels0, labels1], axis=0)
        label = torch.from_numpy(label).contiguous()

        return label