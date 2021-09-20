from dl_toolbox.torch_datasets import OneImage
import torch
import numpy as np

class SemcityBdsdDs(OneImage):

    def __init__(self, *args, **kwargs):

        super(SemcityBdsdDs, self).__init__(*args, **kwargs)
        self.labels_desc = [
            (0, (255, 255, 255), 'void'),
            (1, (38, 38, 38), 'impervious surface'),
            (2, (238, 118, 33), 'building'),
            (3, (34, 139, 34), 'pervious surface'),
            (4, (0, 222, 137), 'high vegetation'),
            (5, (255, 0, 0), 'car'),
            (6, (0, 0, 238), 'water'),
            (7, (160, 30, 230), 'sport venues')
        ]

    def rgb_to_onehot(self, rgb_label):

        onehot_masks = []
        for _, color, _ in self.labels_desc:
            d = rgb_label[0, :, :] == color[0]
            d = np.logical_and(d, (rgb_label[1, :, :] == color[1]))
            d = np.logical_and(d, (rgb_label[2, :, :] == color[2]))
            onehot_masks.append(d.astype(float))
        onehot = np.stack(onehot_masks, axis=0)
        onehot = torch.from_numpy(onehot).contiguous()

        return onehot

    def label_to_rgb(self, label):

        rgb_label = np.zeros(shape=(*label.shape, 3), dtype=float)
        for val, color, _ in self.labels_desc:
            mask = np.array(label == val)
            rgb_label[mask] = np.array(color)
        rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

        return torch.from_numpy(rgb_label).float()

    def process_image(self, image):

        return torch.from_numpy(image[[3,2,1], :, :]).contiguous()

    def process_label(self, label):

        return self.rgb_to_onehot(label)
