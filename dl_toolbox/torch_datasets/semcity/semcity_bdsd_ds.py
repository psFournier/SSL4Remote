from dl_toolbox.torch_datasets import OneImage
import torch
import numpy as np

class SemcityBdsdDs(OneImage):

    def __init__(self, *args, **kwargs):

        super(SemcityBdsdDs, self).__init__(*args, **kwargs)
        self.labels_desc = [
            (0, (255, 255, 255), 'void', 923176),
            (1, (38, 38, 38), 'impervious surface', 8214402),
            (2, (238, 118, 33), 'building', 12857668),
            (3, (34, 139, 34), 'pervious surface', 13109372),
            (4, (0, 222, 137), 'high vegetation', 1825718),
            (5, (255, 0, 0), 'car', 9101418),
            (6, (0, 0, 238), 'water', 1015653),
            (7, (160, 30, 230), 'sport venues', 1335825)
        ]
        # Min and max are 1 and 99 percentiles
        self.image_stats = {
            'num_channels': 8,
            'min' : np.array([245,166,167,107,42,105,60,48]),
            'max' : np.array([615, 681, 1008, 1087, 732, 1065, 1126, 1046])
        }

    def rgb_to_onehot(self, rgb_label):

        onehot_masks = []
        for _, color, _, _ in self.labels_desc:
            d = rgb_label[0, :, :] == color[0]
            d = np.logical_and(d, (rgb_label[1, :, :] == color[1]))
            d = np.logical_and(d, (rgb_label[2, :, :] == color[2]))
            onehot_masks.append(d.astype(float))
        onehot = np.stack(onehot_masks, axis=0)
        onehot = torch.from_numpy(onehot).contiguous()

        return onehot

    def label_to_rgb(self, label):

        rgb_label = np.zeros(shape=(*label.shape, 3), dtype=float)
        for val, color, _, _ in self.labels_desc:
            mask = np.array(label == val)
            rgb_label[mask] = np.array(color)
        rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

        return torch.from_numpy(rgb_label).float()

    def process_image(self, image):

        out = image[[3, 2, 1], :, :]
        min = self.image_stats['min']
        max = self.image_stats['max']
        for i, channel in enumerate([3,2,1]):
            out[i, :, :] = np.clip(((out[i, :, :] - min[channel]) / (max[channel] - min[channel])), 0, 1) * 255
        out = torch.from_numpy(out).contiguous()

        return out

    def process_label(self, label):

        return self.rgb_to_onehot(label)
