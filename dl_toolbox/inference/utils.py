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
import dl_toolbox.augmentations as aug 
import pandas as pd
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from dl_toolbox.torch_datasets.utils import *
from functools import partial

anti_t_dict = {
    'hflip': 'hflip',
    'vflip': 'vflip',
    'd1flip': 'd1flip',
    'd2flip': 'd2flip',
    'rot90': 'rot270',
    'rot180': 'rot180',
    'rot270': 'rot90'
}


def probas_to_preds(probas):

    return torch.argmax(probas, dim=1)

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
    dataset,
    module,
    batch_size,
    workers,
    tta,
    mode
):
    
    device = module.device

    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=CustomCollate(batch_aug='no'),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=worker_init_function
    )

    num_classes = module.num_classes - int(not module.train_with_void)
    pred_sum = torch.zeros(size=(num_classes, dataset.tile.height, dataset.tile.width))
    mask_sum = torch.zeros(size=(dataset.tile.height, dataset.tile.width))

    def dist_to_edge(i, j, h, w):

        mi = np.minimum(i+1, h-i)
        mj = np.minimum(j+1, w-j)
        return np.minimum(mi, mj)

    crop_mask = np.fromfunction(
        function=partial(
            dist_to_edge,
            h=dataset.crop_size,
            w=dataset.crop_size
        ),
        shape=(dataset.crop_size, dataset.crop_size),
        dtype=int
    )
    crop_mask = torch.from_numpy(crop_mask).float()

    for i, batch in enumerate(dataloader):
        
        print('batch ', i)
        inputs, labels, windows = batch['image'], batch['mask'], batch['window']

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
                w.row_off-dataset.tile.row_off:w.row_off-dataset.tile.row_off+w.height,
                w.col_off-dataset.tile.col_off:w.col_off-dataset.tile.col_off+w.width
            ] += prob
            mask_sum[
                w.row_off-dataset.tile.row_off:w.row_off-dataset.tile.row_off+w.height,
                w.col_off-dataset.tile.col_off:w.col_off-dataset.tile.col_off+w.width
            ] += 1
                
    probas = torch.div(pred_sum, mask_sum)

    return probas.detach().cpu().numpy()

def batch_forward(inputs, module, tta=None):
    
    if tta:
        inputs, _ = aug_dict[tta](p=1)(inputs)
    with torch.no_grad():
        outputs = module.forward(inputs.to(module.device)).cpu()
    if tta and tta in anti_t_dict:
        outputs, _ = aug_dict[anti_t_dict[tta]](p=1)(outputs)

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

def cm2metrics(cm, ignore_index=-1):
        """Compute metrics directly from a confusion matrix.
        Return per-class and average metrics.
        Per-class metrics: 
            F1 score, Recall, Precision, IoU
        Average metrics: 
            Overall Accuracy (micro average), Kappa score, 
            macro averages for F1, Recall, Precision and IoU.
        """ 
        labels = np.arange(cm.shape[0])
        TP = np.array([cm[i, i] for i in labels if i != ignore_index])
        FN = np.array([
            sum([cm[i, j] for j in labels if j!=i]) for i in labels if i != ignore_index])
        FP = np.array([
            sum([cm[i, j] for i in labels if i!=j]) for j in labels if j != ignore_index])
            
        f1 = (2 * TP) / (2 * TP + FP + FN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        iou = TP / (TP + FN + FP)

        metrics_per_class = pd.DataFrame({
            "F1": f1,
            "Recall": recall,
            "Precision": precision,
            "IoU": iou
        })

        #mF1 = (2*np.sum(TP)) / (2 * np.sum(TP) + np.sum(FP) +np.sum(FN))
        #mRecall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
        #mPrecision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
        #mIou = np.sum(TP) / (np.sum(TP) + np.sum(FP) + np.sum(FN))

        # Accuracy
        obs_acc = sum(TP) / np.sum(cm[np.array([l for l in labels if l != ignore_index]), :])
        # KAPPA
        marg_freq = np.sum(cm, axis=0) * np.sum(cm, axis=1) / np.sum(cm)
        exp_acc = sum(marg_freq) / np.sum(cm)
        kappa = (obs_acc - exp_acc) / (1 - exp_acc)
        #average_metrics = pd.DataFrame({
        #    "mF1": [mF1],
        #    "mRecall": [mRecall],
        #    "mPrecision": [mPrecision],
        #    "mIoU": [mIou],
        #    "OAccuracy": [obs_acc],
        #    "Kappa": [kappa]
        #})
       
        average_metrics = pd.DataFrame({
            "mF1": [np.mean(f1)],
            "mRecall": [np.mean(recall)],
            "mPrecision": [np.mean(precision)],
            "mIoU": [np.mean(iou)],
            "OAccuracy": [obs_acc],
            "Kappa": [kappa]
        })
        return metrics_per_class, average_metrics

def compute_cm(
    preds,
    labels,
    num_classes
):
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
    cm = np.zeros(shape=(num_classes, num_classes))
    for window in windows:        
        window_labels = rasterio.open(label_path).read(window=window, out_dtype=np.uint8)
        window_labels = raw_labels_to_labels(window_labels, dataset=dataset_type)
        window_preds = preds[
            window.row_off-row_off:window.row_off-row_off+window.height, 
            window.col_off-col_off:window.col_off-col_off+window.width
        ]

        cm += confusion_matrix(
            window_labels.numpy().flatten(),
            window_preds.numpy().flatten(),
            labels = np.arange(num_classes)
        )

    return cm


#def compute_metrics(
#    preds,
#    label_path,
#    dataset_type,
#    tile,
#    eval_with_void,
#    num_classes
#):
#
#    ignore_index = None if eval_with_void else 0
#    metrics = {
#        'accuracy': [],
#        'f1': [],
#        'iou': [],
#        'f1_per_class': [],
#        'accu_per_class': [],
#    }
#    col_off, row_off, width, height = tile
#    windows = get_tiles(
#        nols=width, 
#        nrows=height, 
#        size=width, 
#        size2=height,
#        step=width,
#        step2=height,
#        row_offset=row_off, 
#        col_offset=col_off
#    )
#    for window in windows:        
#        window_labels = rasterio.open(label_path).read(window=window, out_dtype=np.uint8)
#        window_labels = raw_labels_to_labels(window_labels, dataset=dataset_type)
#        window_preds = preds[
#            window.row_off-row_off:window.row_off-row_off+window.height, 
#            window.col_off-col_off:window.col_off-col_off+window.width
#        ]
#        accuracy = M.accuracy(torch.unsqueeze(window_preds, 0),
#                              torch.unsqueeze(window_labels, 0),
#                              ignore_index=ignore_index)
#        accu_per_class = M.accuracy(torch.unsqueeze(window_preds, 0),
#                              torch.unsqueeze(window_labels, 0),
#                              ignore_index=ignore_index,
#                              average='none',
#                              num_classes=num_classes)
# 
#        f1 = M.f1_score(
#            torch.unsqueeze(window_preds, 0),
#            torch.unsqueeze(window_labels, 0),
#            ignore_index=ignore_index,
#            mdmc_average="global"
#        )
#        f1_per_class = M.f1_score(
#            torch.unsqueeze(window_preds, 0),
#            torch.unsqueeze(window_labels, 0),
#            ignore_index=ignore_index,
#            average='none',
#            mdmc_average='global',
#            num_classes=num_classes
#        )
#        iou = M.jaccard_index(
#            torch.unsqueeze(window_preds, 0),
#            torch.unsqueeze(window_labels, 0),
#            ignore_index=ignore_index
#        )
#        metrics['iou'].append(iou)
#        metrics['accuracy'].append(accuracy)
#        metrics['f1'].append(f1)
#        metrics['accu_per_class'].append(torch.nan_to_num(accu_per_class)) 
#        metrics['f1_per_class'].append(torch.nan_to_num(f1_per_class))
#    
#    ret = {
#        'accuracy': np.mean(metrics['accuracy']), 
#        'f1': np.mean(metrics['f1']),
#        'iou': np.mean(metrics['iou'])
#    }
#    return metrics
 

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

#def save_metrics(self, results, out_dir):
#    """
#    Save metrics in a single Excel file, saved in /out_dir/metrics.xlsx.
#    Also compute the normalized confusion matrices and save them in /out_dir.
#    """
#    metrics_out_path = os.path.join(out_dir, "metrics.xlsx")
#    writer = pd.ExcelWriter(metrics_out_path, engine="xlsxwriter")
#    
#    # Log average metrics
#    average_metrics = [results[city]['average_metrics'] for city in results]
#    average_metrics = pd.concat(average_metrics)
#    average_metrics.loc['mean'] = average_metrics.mean()
#    bold_idxs = [average_metrics.index.get_loc('mean')]
#    if self.save_std:
#        average_metrics.loc['std'] = average_metrics.std() 
#        bold_idxs.append(average_metrics.index.get_loc('std'))      
#    average_metrics.to_excel(writer, sheet_name="Summary")
#
#    workbook  = writer.book
#    worksheet = writer.sheets['Summary']
#
#    # Write statistics lines in bold
#    cell_format = workbook.add_format({
#        'bold': True,
#        'text_wrap': True,
#        'valign': 'top',
#        })
#    for row, col in product(bold_idxs, range(len(average_metrics.columns))):
#        val = average_metrics.iloc[row, col]
#        worksheet.write(row+1, col+1, val, cell_format)
#
#    # Log metrics per class
#    cities = results.keys()
#    metrics_per_class = [results[city]['metrics_per_class'] for city in results]
#    metrics_per_class = pd.concat(metrics_per_class)
#    for col in metrics_per_class.columns:
#        col_df = metrics_per_class[col]
#        metric = [col_df.xs(key=c) for c in cities]
#        metric = pd.concat(metric, axis=1)
#        metric.rename(index=self.code2name)
#        metric = metric.set_axis(cities, axis=1)
#        
#        metric.to_excel(writer, sheet_name=col)
#
#    # Log confusion matrices
#    conf_mats = [results[city]['cm'] for city in results]
#    for cm, city in zip(conf_mats, cities):
#        cm_df = self.cm2df(cm)
#        cm_df.to_excel(writer, sheet_name=f"{city}_conf_mat")
#        self.plot_matrix(
#            cm=cm, 
#            out_path=os.path.join(out_dir, f"{city}_conf_mat.png"),
#            normalisation='true')
#
#    writer.save()
#
#def cm2df(cm):
#    """Convert a confusion matrix into a Pandas DataFrame, labeling the rows and 
#    columns using the nomenclature.
#    """
#    cm_df = pd.DataFrame(cm)
#    columns = [('Prediction', label) for label in self.code2name.values()]
#    lines = [('True Label', label) for label in self.code2name.values()]
#    cm_df.columns = pd.MultiIndex.from_tuples(columns)
#    cm_df.index = pd.MultiIndex.from_tuples(lines)
#
#    return cm_df
#
#
#
#def plot_matrix(self, cm, out_path, normalisation=None):
#    """Make a PNG figure of a confusion matrix."""
#    if normalisation=="true":
#        cmf=cm/(np.sum(cm, axis=1)[:,None])
#    elif normalisation=="pred":
#        cmf=cm/(np.sum(cm, axis=0)[:,None])
#    else:
#        cmf=cm
#
#    labels = self.code2name.values()
#    fig, ax = plt.subplots(figsize=(15, 15))
#    disp = ConfusionMatrixDisplay(
#        confusion_matrix=cmf, 
#        display_labels=labels)
#    disp.plot(
#        include_values=labels,
#        cmap=plt.cm.Blues, ax=ax, xticks_rotation=45,
#        values_format=None)
#    plt.savefig(out_path)

