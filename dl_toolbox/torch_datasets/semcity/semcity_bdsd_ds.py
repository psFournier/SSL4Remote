from dl_toolbox.torch_datasets import OneImage
import torch
import numpy as np

class SemcityBdsdDs(OneImage):

    labels_desc = [
        (0, (255, 255, 255), 'void', 1335825),
        (1, (38, 38, 38), 'impervious surface', 13109372),
        (2, (238, 118, 33), 'building', 9101418),
        (3, (34, 139, 34), 'pervious surface', 12857668),
        (4, (0, 222, 137), 'high vegetation', 8214402),
        (5, (255, 0, 0), 'car', 1015653),
        (6, (0, 0, 238), 'water', 923176),
        (7, (160, 30, 230), 'sport venues', 1825718)
    ]
    # Min and max are 1 and 99 percentiles
    image_stats = {
        'num_channels': 8,
        'min' : np.array([245,166,167,107,42,105,60,48]),
        'max' : np.array([615, 681, 1008, 1087, 732, 1065, 1126, 1046])
    }

    def __init__(self, *args, **kwargs):

        super(SemcityBdsdDs, self).__init__(*args, **kwargs)

    @classmethod
    def rgb_to_onehot(cls, rgb_label):

        onehot_masks = []
        for _, color, _, _ in cls.labels_desc:
            d = rgb_label[0, :, :] == color[0]
            d = np.logical_and(d, (rgb_label[1, :, :] == color[1]))
            d = np.logical_and(d, (rgb_label[2, :, :] == color[2]))
            onehot_masks.append(d.astype(float))
        onehot = np.stack(onehot_masks, axis=0)

        return onehot

    @classmethod
    def label_to_rgb(cls, label):

        rgb_label = torch.zeros(size=(*label.shape, 3), device=label.device)
        for val, color, _, _ in cls.labels_desc:
            mask = label == val
            rgb_label[mask] = torch.tensor(color, device=label.device, dtype=torch.float32)
        rgb_label = rgb_label.permute(0, 3, 1, 2)

        return rgb_label

    def process_image(self, image):

        out = image[[3, 2, 1], :, :]
        min = self.image_stats['min']
        max = self.image_stats['max']
        for i, channel in enumerate([3,2,1]):
            out[i, :, :] = np.clip(((out[i, :, :] - min[channel]) / (max[channel] - min[channel])), 0, 1)
        out = torch.from_numpy(out).contiguous()

        # return torch.from_numpy(image[[3, 2, 1], :, :]).contiguous()
        return out

    def process_label(self, label):

        return torch.from_numpy(self.rgb_to_onehot(label)).contiguous()
