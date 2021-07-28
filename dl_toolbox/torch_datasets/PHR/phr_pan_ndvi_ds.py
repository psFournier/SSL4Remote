import imagesize
import rasterio
import numpy as np
import torch

from dl_toolbox.torch_datasets import OneImage


class PhrPanNdviDs(OneImage):

    def __init__(self, ndvi_path, *args, **kwargs):

        super(PhrPanNdviDs, self).__init__(*args, **kwargs)
        assert imagesize.get(self.image_path) == imagesize.get(ndvi_path)
        self.pan_path = self.image_path
        self.ndvi_path = ndvi_path

    def process_image(self, image):

        return torch.from_numpy(image).contiguous()

    def process_label(self, label):

        labels0 = np.zeros(shape=label.shape[1:], dtype=float)
        labels1 = np.zeros(shape=label.shape[1:], dtype=float)
        mask = label[0, :, :] == 5
        np.putmask(labels0, ~mask, 1.)
        np.putmask(labels1, mask, 1.)
        label = np.stack([labels0, labels1], axis=0)
        label = torch.from_numpy(label).contiguous()

        return label

    def __getitem__(self, idx):

        tile_idx = self.idxs[idx]
        window = self.get_window(tile_idx)

        with rasterio.open(self.image_path) as pan_file:
            pan = pan_file.read(window=window, out_dtype=np.float32)
            pan = self.process_image(pan)

        with rasterio.open(self.ndvi_path) as ndvi_file:
            ndvi = ndvi_file.read(window=window, out_dtype=np.float32)
            ndvi[np.isnan(ndvi)] = 0
            ndvi[np.isinf(ndvi)] = 0
            ndvi = self.process_image(ndvi)

        image = np.concatenate([pan, ndvi], axis=0)

        label = None
        if self.label_path is not None:
            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                label = self.process_label(label)

        if self.transforms is not None:
            end_image, end_mask = self.transforms(img=image, label=label)
        else:
            end_image, end_mask = image, label

        return {'orig_image': image, 'orig_mask': label, 'image': end_image, 'window': window, 'mask': end_mask}
