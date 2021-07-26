import imagesize
import rasterio
import numpy as np
import torch

from dl_toolbox.torch_datasets import OneImage


class PhrPanNdvi(OneImage):

    def __init__(self, ndvi_path, *args, **kwargs):

        super(PhrPanNdvi, self).__init__(*args, **kwargs)
        assert imagesize.get(self.image_path) == imagesize.get(ndvi_path)
        self.pan_path = self.image_path
        self.ndvi_path = ndvi_path

    def __getitem__(self, idx):

        tile_idx = self.idxs[idx]
        window = self.get_window(tile_idx)

        with rasterio.open(self.image_path) as pan_file:
            pan = pan_file.read(window=window, out_dtype=np.float32)
            pan = torch.from_numpy(pan).contiguous() / 255

        with rasterio.open(self.ndvi_path) as ndvi_file:
            ndvi = ndvi_file.read(window=window, out_dtype=np.float32)
            ndvi[np.isnan(ndvi)] = 0
            ndvi[np.isinf(ndvi)] = 0

        image = np.concatenate([pan, ndvi], axis=0)

        mask = None
        if self.label_path is not None:
            with rasterio.open(self.label_path) as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)
                mask = self.label_formatter(label)
                mask = torch.from_numpy(mask).contiguous()

        if self.transforms is not None:
            end_image, end_mask = self.transforms(img=image, label=mask)
        else:
            end_image, end_mask = image, mask

        return {'orig_image': image, 'orig_mask': mask, 'image': end_image, 'window': window, 'mask': end_mask}
