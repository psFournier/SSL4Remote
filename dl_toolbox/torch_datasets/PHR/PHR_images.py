from dlcooker.forward.datasets.one_image import OneImage
import imagesize
import rasterio
import numpy as np

class PHR_PAN(OneImage):

    def __init__(self, pansharpened_fname, *args, **kwargs):

        super(PHR_PAN, self).__init__(image_path=pansharpened_fname, *args, **kwargs)

class PHR_PAN_NDVI(OneImage):

    def __init__(self, pan_fname, ndvi_fname, *args, **kwargs):

        assert imagesize.get(pan_fname) == imagesize.get(ndvi_fname)
        super(PHR_PAN_NDVI, self).__init__(image_path=pan_fname, *args, **kwargs)
        self.pan_image_path = pan_fname
        self.ndvi_image_path = ndvi_fname

    def __getitem__(self, idx):

        pan_image_dict = super(PHR_PAN_NDVI, self).__getitem__(idx)
        window = pan_image_dict['window']
        with rasterio.open(self.ndvi_image_path) as ndvi_image_file:

            ndvi_image = ndvi_image_file.read(window=window, out_dtype=np.float32).transpose((1, 2, 0))
            ndvi_image[np.isnan(ndvi_image)] = 0
            ndvi_image[np.isinf(ndvi_image)] = 0
            if self.transforms is not None:
                ndvi_image = self.transforms(image = ndvi_image)['image']

        image = np.concatenate([pan_image_dict['image'], ndvi_image], axis = 2)

        return {'image': image, 'window': window}
