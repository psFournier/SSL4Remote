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
    'semcity': SemcityBdsdDs,
    'digitanie': DigitanieDs
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

def raw_labels_to_labels(labels, dataset=None):

    if dataset=='semcity':
        return rgb_to_labels(labels, dataset=dataset)
    elif dataset=='digitanie':
        return torch.squeeze(torch.from_numpy(labels)).long()
    else:
        raise NotImplementedError

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
    tta,
    mode
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

    num_classes = module.num_classes - int(not module.train_with_void)
    pred_sum = torch.zeros(size=(num_classes, window.height, window.width))
    mask_sum = torch.zeros(size=(window.height, window.width))

    for i, batch in enumerate(dataloader):
        
        print('batch ', i)
        inputs, _, windows = batch['image'], batch['mask'], batch['window']

        outputs = batch_forward(inputs, module)
        window_list = windows[:]

        for t in tta:

            outputs_tta = batch_forward(inputs, module, t)
            outputs = torch.vstack([outputs, outputs_tta])
            window_list += windows[:]

        split_pred = torch.split(outputs, 1, dim=0)
        pred_list = [torch.squeeze(e, dim=0) for e in split_pred]
        
        for pred, w in zip(pred_list, window_list):
            if mode=='softmax':
                prob = pred.softmax(dim=0)
            elif mode=='sigmoid':
                prob = torch.sigmoid(pred)
            pred_sum[
                :, 
                w.row_off-window.row_off:w.row_off-window.row_off+w.height,
                w.col_off-window.col_off:w.col_off-window.col_off+w.width
            ] += prob
            mask_sum[
                w.row_off-window.row_off:w.row_off-window.row_off+w.height,
                w.col_off-window.col_off:w.col_off-window.col_off+w.width
            ] += 1
                
    probas = torch.div(pred_sum, mask_sum)

    return probas

def batch_forward(inputs, module, tta=None):
    
    if tta:
        inputs, _ = image_level_aug[tta](p=1)(inputs)
    with torch.no_grad():
        outputs = module.forward(inputs.to(module.device)).cpu()
    if tta and tta in anti_t_dict:
        outputs, _ = image_level_aug[anti_t_dict[tta]](p=1)(outputs)

    return outputs

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
 
def write_array(inputs, tile, output_path, profile):

    window = get_window(tile)
    transform = get_window_transform(
        window,
        profile
    )
    new_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': window.width,
        'height': window.height,
        'count': inputs.shape[0],
        'crs': profile['crs'],
        'transform': transform,
        'tiled': False,
        'interleave': 'pixel'
    }
    with rasterio.open(output_path, 'w', **new_profile) as dst:
        dst.write(inputs)


def read_probas(input_path):

    with rasterio.open(input_path) as f:
        probas = f.read(out_dtype=np.float32)

    return probas



def compute_metrics(
    preds,
    label_path,
    dataset_type,
    tile,
    eval_with_void
):

    ignore_index = None if eval_with_void else 0
    metrics = {
        'accuracy': [],
        'f1': []
    }
    col_off, row_off, width, height = tile
    windows = get_tiles(
        nols=width, 
        nrows=height, 
        size=width, 
        size2=height,
        step=width,
        step2=height,
        row_offset=row_off, 
        col_offset=col_off
    )
    for window in windows:        
        window_labels = rasterio.open(label_path).read(window=window, out_dtype=np.uint8)
        window_labels = raw_labels_to_labels(window_labels, dataset=dataset_type)
        window_preds = preds[
            window.row_off-row_off:window.row_off-row_off+window.height, 
            window.col_off-col_off:window.col_off-col_off+window.width
        ]
        accuracy = M.accuracy(torch.unsqueeze(window_preds, 0),
                              torch.unsqueeze(window_labels, 0),
                              ignore_index=ignore_index)
        f1 = M.f1_score(
            torch.unsqueeze(window_preds, 0),
            torch.unsqueeze(window_labels, 0),
            ignore_index=ignore_index,
            mdmc_average="global"
        )
        metrics['accuracy'].append(accuracy)
        metrics['f1'].append(f1)
    
    
    ret = {'accuracy': np.mean(metrics['accuracy']), 'f1': np.mean(metrics['f1'])}
    return ret
 

def visualize_errors(
    preds,
    label_path,
    dataset_type,
    output_path,
    tile,
    initial_profile,
    class_id=None,
    eval_with_void=False
):
    
    window = get_window(tile)

    with rasterio.open(label_path) as f:
        label_tile = f.read(window=window, out_dtype=np.uint8)
    label_tile = raw_labels_to_labels(label_tile, dataset=dataset_type)

    overlay = np.zeros(shape=(window.height, window.width, 3), dtype=np.uint8)
    if class_id:
        label_bool = label_tile == class_id
        pred_bool = preds == class_id

        # Correct predictions (Hits) painted with green
        overlay[label_bool & pred_bool] = np.array([0, 250, 0], dtype=overlay.dtype)
        # Misses painted with red
        overlay[label_bool & ~pred_bool] = np.array([250, 0, 0], dtype=overlay.dtype)
        # False alarm painted with yellow
        overlay[~label_bool & pred_bool] = np.array([250, 250, 0], dtype=overlay.dtype)
    else:
        if not eval_with_void:
            void_bool = label_tile == 0
        else:
            void_bool = np.zeros(shape=(window.height, window.width), dtype=np.uint8)
        correct = label_tile == preds
        print(correct.shape)
        overlay[correct & ~void_bool] = np.array([0,250,0], dtype=overlay.dtype)
        overlay[~correct & ~void_bool] = np.array([250,0,0], dtype=overlay.dtype)

    write_array(overlay.transpose(2,0,1), tile, output_path, initial_profile)
