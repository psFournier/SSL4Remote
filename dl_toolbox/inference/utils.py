from torch.utils.data import DataLoader, ConcatDataset
from argparse import ArgumentParser
import torch
import imagesize
import numpy as np
import rasterio
from rasterio.windows import Window
from torch import nn
from dl_toolbox.torch_datasets import SemcityBdsdDs, DigitanieDs
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.lightning_modules import Unet
from dl_toolbox.utils import worker_init_function, get_tiles
import torchmetrics.functional as  M
from dl_toolbox.augmentations import image_level_aug

anti_t_dict = {
    'hflip': 'hflip',
    'vflip': 'vflip',
    'd1flip': 'd1flip',
    'd2flip': 'd2flip',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90'
}

datasets = {
    'semcity': SemcityBdsdDs 
}


def probas_to_preds(probas):

    return torch.argmax(probas, dim=1)

def label_to_rgb(label, color_map):

    rgb_label = np.zeros(shape=(*labels.shape, 3), dtype=float)
    for val, color in color_map.items():
        mask = np.array(labels == val)
        rgb_label[mask] = np.array(color)
    rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

    return rgb_label



def compute_probas(
    image_path,
    dataset_type,
    tile,
    module,
    crop_size,
    crop_step,
    batch_size,
    workers,
    tta
):
    
    device = module.device

    col_off, row_off, width, height = tile
    window = Window(
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height
    )

    dataset = datasets[dataset_type](
        image_path=image_path,
        fixed_crops=True,
        tile=window,
        crop_size=crop_size,
        crop_step=crop_step,
        img_aug='no'
    )

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=CustomCollate(),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )


    pred_sum = torch.zeros(size=(module.num_classes, height, width))

    for batch in dataloader:

        inputs, _, windows = batch['image'], batch['mask'], batch['window']
        
        with torch.no_grad():
            outputs = module.forward(inputs.to(device)).cpu()

        split_pred = np.split(outputs, outputs.shape[0], axis=0)
        pred_list = [np.squeeze(e, axis=0) for e in split_pred]
        window_list = windows[:]
        
        for pred, window in zip(pred_list, window_list):
            pred_sum[:, window.row_off:window.row_off + window.width, window.col_off:window.col_off + window.height] += pred
    
    for t_name in tta:
        for batch in dataloader:
            inputs = batch['image']
            aug_inputs, _ = image_level_aug[t_name](p=1)(inputs)
            with torch.no_grad():
                outputs = module.forward(aug_inputs.to(device)).cpu()
            if t_name in anti_t_dict:
                outputs, _ = image_level_aug[anti_t_dict[t_name]](p=1)(outputs)
            split_pred = np.split(outputs, outputs.shape[0], axis=0)
            pred_list = [np.squeeze(e, axis=0) for e in split_pred]
            window_list = batch['window'][:]
            
            for pred, window in zip(pred_list, window_list):
                pred_sum[:, 
                        window.row_off:window.row_off + window.width, 
                        window.col_off:window.col_off + window.height] += pred

    probas = pred_sum.softmax(dim=0)

    return probas

def write_rgb_pred(rgb_preds, output_path, initial_profile):

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': initial_profile['width'],
        'height': initial_profile['height'],
        'count': 3,
        'crs': initial_profile['crs'],
        'transform': initial_profile['transform'],
        'tiled': False,
        'interleave': 'pixel'
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(3):
            dst.write(rgb_preds[i, :, :], indexes=i+1)



def write_probas(probas, output_path, initial_profile):

    num_classes = probas.shape[0]
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': initial_profile['width'],
        'height': initial_profile['height'],
        'count': num_classes,
        'crs': initial_profile['crs'],
        'transform': initial_profile['transform'],
        'tiled': False,
        'interleave': 'pixel'
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(num_classes):
            dst.write(probas[i, :, :], indexes=i+1)

def read_probas(input_path):

    with rasterio.open(input_path) as f:
        probas = f.read(out_dtype=np.float32)

    return probas



def compute_metrics(
    preds,
    label_path,
    tile
):

    metrics = {
        'accuracy': []
    }
    col_off, row_off, width, height = tile
    windows = get_tiles(
        nols=width, 
        nrows=height, 
        size=256, 
        step=256,
        row_offset=row_off, 
        col_offset=col_off
    )
    for window in windows:        
        window_labels = rasterio.open(label_path).read(window=window, out_dtype=np.uint8)
        window_labels = torch.from_numpy(window_labels)
        window_preds = preds[
            window.row_off-row_off:window.row_off-row_off+window.width, 
            window.col_off-col_off:window.col_off-col_off+window.height
        ]
        accuracy = M.accuracy(torch.unsqueeze(window_preds, 0),
                              torch.unsqueeze(window_labels, 0),
                              ignore_index=0)
        metrics['accuracy'].append(accuracy)

    return {'accuracy': np.mean(metrics['accuracy'])}
 

def visualize_errors(
    preds,
    label_path,
    class_id,
    output_path,
    tile,
    initial_profile
):
    

    col_off, row_off, width, height = tile
    window = Window(
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height
    )

    with rasterio.open(label_path) as f:
        label_tile = f.read(window=window, out_dtype=np.uint8)

    label_bool = np.squeeze(label_tile == class_id)
    pred_bool = np.squeeze(np.array(preds) == class_id)
    overlay = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    # Correct predictions (Hits) painted with green
    overlay[label_bool & pred_bool] = np.array([0, 250, 0], dtype=overlay.dtype)
    # Misses painted with red
    overlay[label_bool & ~pred_bool] = np.array([250, 0, 0], dtype=overlay.dtype)
    # False alarm painted with yellow
    overlay[~label_bool & pred_bool] = np.array([250, 250, 0], dtype=overlay.dtype)

    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'nodata': None,
        'width': initial_profile['width'],
        'height': initial_profile['height'],
        'count': 3,
        'crs': initial_profile['crs'],
        'transform': initial_profile['transform'],
        'tiled': False,
        'interleave': 'pixel'
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(overlay.transpose(2,0,1))
