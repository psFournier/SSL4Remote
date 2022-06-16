import os
import csv

from torch.utils.data import ConcatDataset
from rasterio.windows import Window
from dl_toolbox.torch_datasets import *

dataset_dict = {
    'Toulouse': DigitanieToulouseDs,
    'Biarritz': DigitanieBiarritzDs,
    'Montpellier': DigitanieMontpellierDs,
    'Paris': DigitanieParisDs,
    'Strasbourg': DigitanieStrasbourgDs
}

def read_splitfile(
    splitfile,
    data_path,
    train_folds,
    test_folds
):

    test_datasets_args, train_datasets_args = [], []

    reader = csv.reader(splitfile)
    next(reader)
    for row in reader:
              
        city, _, image_path, label_path, x0, y0, w, h, fold = row[:9]
        is_train = int(fold) in train_folds 
        is_test = int(fold) in test_folds
        window = Window(
            col_off=int(x0),
            row_off=int(y0),
            width=int(w),
            height=int(h)
        )

        kwargs = {
            'image_path':os.path.join(data_path, image_path),
            'fixed_crops':is_test,
            'tile':window,
        }

        try:
            orig_img = row[9]
            kwargs['full_raster_path'] = os.path.join(data_path, orig_img)
        except:
            pass

        if label_path != 'none': 
            kwargs['label_path'] = os.path.join(data_path, label_path)
 
        if is_train:
            train_datasets_args.append((dataset_dict[city], kwargs))
        elif is_test:
            test_datasets_args.append((dataset_dict[city], kwargs))
        else:
            pass

    return train_datasets_args, test_datasets_args


def build_split_from_csv(
    splitfile,
    data_path,
    dataset_cls,
    img_aug,
    crop_size,
    one_hot,
    train_folds,
    test_folds
):
    test_datasets, train_datasets = [], []
    test_set, train_set = None, None

    reader = csv.reader(splitfile)
    next(reader)
    for row in reader:
              
        city, _, image_path, label_path, x0, y0, w, h, fold = row[:9]
        is_train = int(fold) in train_folds 
        is_test = int(fold) in test_folds
        aug = 'no' if is_test else img_aug
        window = Window(
            col_off=int(x0),
            row_off=int(y0),
            width=int(w),
            height=int(h)
        )

        kwargs = {
            'image_path':os.path.join(data_path, image_path),
            'fixed_crops':is_test,
            'tile':window,
            'crop_size':crop_size,
            'crop_step':crop_size,
            'img_aug':aug,
            'one_hot':one_hot
        }

        try:
            orig_img = row[9]
            kwargs['full_raster_path'] = os.path.join(data_path, city, orig_img)
        except:
            pass

        if label_path != 'none': 
            kwargs['label_path'] = os.path.join(data_path, label_path)

        dataset = dataset_cls(**kwargs)
            
        if is_train:
            train_datasets.append(dataset)
        elif is_test:
            test_datasets.append(dataset)
        else:
            pass

    return train_datasets, test_datasets
