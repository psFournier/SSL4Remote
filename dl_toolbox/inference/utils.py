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

def labels_to_rgb(labels, color_map=None, dataset=None):

    assert color_map or dataset
    if not color_map:
        color_map=datasets[dataset].color_map
    rgb_label = np.zeros(shape=(*labels.shape, 3), dtype=float)
    for val, color in color_map.items():
        mask = np.array(labels == val)
        rgb_label[mask] = np.array(color)
    rgb_label = np.transpose(rgb_label, axes=(0, 3, 1, 2))

    return rgb_label

def rgb_to_labels(rgb, color_map=None, dataset=None):

    assert color_map or dataset
    if not color_map:
        color_map=datasets[dataset].color_map
    labels = torch.zeros(size=(rgb.shape[1:]))
    for val, color in color_map.items():
        d = rgb[0, :, :] == color[0]
        d = np.logical_and(d, (rgb[1, :, :] == color[1]))
        d = np.logical_and(d, (rgb[2, :, :] == color[2]))
        labels[d] = val

    return labels.long()

def get_window(tile):

    col_off, row_off, width, height = tile
    window = Window(
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height
    )

    return window

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

    window = get_window(tile)

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


    pred_sum = torch.zeros(size=(module.num_classes, window.height, window.width))

    for batch in dataloader:

        inputs, _, windows = batch['image'], batch['mask'], batch['window']

        outputs = batch_forward(inputs, module)
        window_list = windows[:]

        for t in tta:

            outputs_tta = batch_forward(inputs, module, t)
            outputs = torch.vstack([outputs, outputs_tta])
            window_list += windows[:]

        split_pred = np.split(outputs, outputs.shape[0], axis=0)
        pred_list = [np.squeeze(e, axis=0) for e in split_pred]
        
        for pred, w in zip(pred_list, window_list):
            pred_sum[
                :, 
                w.row_off:w.row_off + w.width,
                w.col_off:w.col_off + w.height
            ] += pred
    
    probas = pred_sum.softmax(dim=0)

    return probas

def batch_forward(inputs, module, tta=None):
    
    if tta:
        inputs, _ = image_level_aug[tta](p=1)(inputs)
    with torch.no_grad():
        outputs = module.forward(inputs.to(module.device)).cpu()
    if tta and tta in anti_t_dict:
        outputs, _ = image_level_aug[anti_t_dict[tta]](p=1)(outputs)

    return outputs

def write_rgb_preds(rgb_preds, tile, output_path, initial_profile):

    window = get_window(tile)
    transform = get_window_transform(
        window,
        initial_profile
    )
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': window.width,
        'height': window.height,
        'count': 3,
        'crs': initial_profile['crs'],
        'transform': transform,
        'tiled': False,
        'interleave': 'pixel'
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(3):
            dst.write(rgb_preds[i, :, :], indexes=i+1)

def get_window_transform(window, profile):

    bounds = rasterio.windows.bounds(
        window,
        transform=profile['transform']
    )
    new_transform = rasterio.transform.from_bounds(
        *bounds,
        window.width,
        window.height
    )

    return new_transform
 

def write_probas(probas, tile, output_path, initial_profile):

    num_classes = probas.shape[0]
    window = get_window(tile)
    transform = get_window_transform(
        window,
        initial_profile
    )
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': window.width,
        'height': window.height,
        'count': num_classes,
        'crs': initial_profile['crs'],
        'transform': transform,
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
    dataset_type,
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
        window_labels = rgb_to_labels(window_labels, dataset=dataset_type)
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
    dataset_type,
    class_id,
    output_path,
    tile,
    initial_profile
):
    
    window = get_window(tile)

    with rasterio.open(label_path) as f:
        label_tile = f.read(window=window, out_dtype=np.uint8)
    label_tile = rgb_to_labels(label_tile, dataset=dataset_type)

    label_bool = label_tile == class_id
    pred_bool = preds == class_id
    overlay = np.zeros(shape=(window.height, window.width, 3), dtype=np.uint8)

    # Correct predictions (Hits) painted with green
    overlay[label_bool & pred_bool] = np.array([0, 250, 0], dtype=overlay.dtype)
    # Misses painted with red
    overlay[label_bool & ~pred_bool] = np.array([250, 0, 0], dtype=overlay.dtype)
    # False alarm painted with yellow
    overlay[~label_bool & pred_bool] = np.array([250, 250, 0], dtype=overlay.dtype)

    transform = get_window_transform(
        window,
        initial_profile
    )
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': window.width,
        'height': window.height,
        'count': 3,
        'crs': initial_profile['crs'],
        'transform': transform,
        'tiled': False,
        'interleave': 'pixel'
    }
 
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(overlay.transpose(2,0,1))
