# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
from augmentations import *
import numpy as np
import torchmetrics.functional as metrics
import rasterio
import imagesize


class WholeImagePred(pl.Callback):

    def __init__(self, tta, save_output_path=None):

        super(WholeImagePred, self).__init__()
        self.tta_metrics = {}
        self.tta = tta
        self.save_output_path = save_output_path
        self.pred_sum = None
        self.nb_in_sum = None

    def on_test_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:

        dataset = pl_module.test_dataloader.dataloader.dataset
        assert len(dataset.labeled_image_paths) == 1
        self.image_path = dataset.labeled_image_paths[0]
        self.label_path = dataset.label_paths[0]
        self.colors_to_labels = dataset.colors_to_labels

        height, width = imagesize.get(self.image_path)
        self.pred_sum = torch.zeros(size=(pl_module.num_classes, height, width))
        self.nb_in_sum = torch.zeros(size=(pl_module.num_classes, height, width))
        pl_module.eval()

    def on_test_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        test_inputs = batch['image'].to(pl_module.device)
        windows = batch['window']

        standard_pred = outputs['preds'].cpu()
        pred_list = [np.squeeze(e, axis=0) for e in np.split(standard_pred, standard_pred.shape[0], axis=0)]
        window_list = windows[:]

        if 'd4' in self.tta:
            for angle in [0,90,270]:
                for ph in [0, 1]:
                    for pv in [0, 1]:
                        for pt in [0, 1]:
                            if any([a!=0 for a in [angle, ph, pv, pt]]) and all([a!=1 for a in [angle, ph, pv, pt]]):
                                t = Compose([
                                    Rotate(p=1, angles=(angle,)),
                                    Hflip(p=ph),
                                    Vflip(p=pv),
                                    Transpose(p=pt)
                                ])
                                aug_inputs = t(test_inputs)[0]
                                aug_pred = pl_module.network(aug_inputs)
                                anti_t = Compose([
                                    Transpose(p=pt),
                                    Vflip(p=pv),
                                    Hflip(p=ph),
                                    Rotate(p=1, angles=(-angle,))
                                ])
                                pred = anti_t(aug_pred)[0].cpu()
                                pred_list += [np.squeeze(e, axis=0) for e in np.split(pred, pred.shape[0], axis=0)]
                                window_list += windows

        for pred, window in zip(pred_list, window_list):
            self.pred_sum[:, window.row_off:window.row_off + window.width,
            window.col_off:window.col_off + window.height] += pred
            self.nb_in_sum[window.row_off:window.row_off + window.width,
            window.col_off:window.col_off + window.height] += 1

    def on_test_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:

        avg_probs = torch.unsqueeze(self.pred_sum.softmax(dim=0), 0)
        label_file = rasterio.open(self.label_path)
        full_image_label = self.colors_to_labels(label_file.read(out_dtype=np.float32))
        test_labels = torch.unsqueeze(torch.argmax(torch.from_numpy(full_image_label), dim=0).long(), 0)
        IoU = metrics.iou(avg_probs,
                          test_labels,
                          reduction='none',
                          num_classes=pl_module.num_classes)
        self.tta_metrics['TTA_IoU_0'] = IoU[0]
        self.tta_metrics['TTA_IoU_1'] = IoU[1]
        self.tta_metrics['TTA_IoU'] = IoU.mean()

        if self.save_output_path is not None:
            pred_profile = label_file.profile
            pred_profile.update(count=1)
            with rasterio.open(self.save_output_path, 'w', **pred_profile) as dst:
                dst.write(np.uint8(np.argmax(self.pred_sum, axis=0)), indexes=1)



