import warnings
import numpy as np
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset
from abc import ABC
from utils import get_tiles

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class Base(Dataset, ABC):

    nb_labeled_images = 0
    nb_unlabeled_images = 0
    labeled_image_paths = []
    unlabeled_image_paths = []
    label_paths = []
    image_size = (0,0)
    pixels_per_class = [1, 1]

    def __init__(self,
                 data_path,
                 idxs,
                 crop,
                 augmentations,
                 fixed_crop=False,
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.data_path = data_path
        self.path_idxs = idxs
        self.crop = crop
        self.augmentations = augmentations
        self.mean_labeled_pixels = []
        self.std_labeled_pixels = []

        self.fixed_crop = fixed_crop
        # Assumes all image are the same size.
        self.precomputed_crops = [
            window for window in get_tiles(
                image_size=self.image_size,
                width=self.crop,
                height=self.crop,
                col_step=self.crop,
                row_step=self.crop
            )
        ]

    def get_random_crop(self):

        cols, rows = self.image_size
        cx = np.random.randint(0, cols - self.crop - 1)
        cy = np.random.randint(0, rows - self.crop - 1)
        w = Window(cx, cy, self.crop, self.crop)

        return w

    def get_image(self, path_idx, crop_idx):

        image_filepath = (self.labeled_image_paths + self.unlabeled_image_paths)[path_idx]

        with rasterio.open(image_filepath) as image_file:

            if self.fixed_crop:
                window = self.precomputed_crops[crop_idx]
            else:
                window = self.get_random_crop()

            image = image_file.read(window=window, out_dtype=np.uint8).transpose(1, 2, 0)

        return image, window

    def get_label(self, path_idx, window):

        label_filepath = self.label_paths[path_idx]

        with rasterio.open(label_filepath) as label_file:

            label = label_file.read(window=window, out_dtype=np.uint8).transpose(1, 2, 0)

        return label

    def __len__(self):

        return len(self.precomputed_crops) * len(self.path_idxs)

    def __getitem__(self, idx):

        raise NotImplementedError


class BaseUnlabeled(Base, ABC):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        path_idx = self.path_idxs[idx % len(self.path_idxs)]
        crop_idx = idx // len(self.path_idxs)
        image, window = self.get_image(path_idx, crop_idx)
        augment = self.augmentations(image=image)

        return augment['image']


class BaseLabeled(BaseUnlabeled, ABC):

    @staticmethod
    def colors_to_labels(labels_color):

        raise NotImplementedError

    def __init__(self, label_merger=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.label_merger = label_merger

    def __getitem__(self, idx):

        path_idx = self.path_idxs[idx % len(self.path_idxs)]
        crop_idx = idx // len(self.path_idxs)
        image, window = self.get_image(path_idx, crop_idx)
        label = self.get_label(path_idx, window)
        mask = self.colors_to_labels(label)
        if self.label_merger is not None:
            mask = self.label_merger(mask)
        augment = self.augmentations(
            image=image,
            mask=mask
        )

        return augment['image'], augment['mask']
