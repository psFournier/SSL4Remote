# Standard libraries
from typing import Any, Optional

# Third-party libraries
from playground_metrics import MeanAveragePrecisionMetric
from pytorch_lightning.metrics import Metric
import rasterio.features
import torch
import numpy as np

class MAPMetric(Metric):

    def __init__(self,
                 threshold: float = 0.5,
                 match_algorithm: str = 'coco',
                 trim_invalid_geometry: bool = True,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):

        # Init parent
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
        )

        # Init mAP computers
        self.computer = MeanAveragePrecisionMetric(
            threshold=threshold,
            match_algorithm=match_algorithm,
            trim_invalid_geometry=trim_invalid_geometry
        )

        # Set attributes
        self.threshold = threshold

    def __repr__(self):
        """Metric name."""
        return f'mAP@{self.threshold}'

    @staticmethod
    def _input_format(preds: torch.Tensor, target: torch.Tensor):
        out = torch.argmax(preds, dim=1).byte()
        out = out.detach().cpu().numpy()
        mask = target.detach().cpu().numpy().astype(np.int32)
        # Extract features from mask
        ground_truths = []
        for shape, value in rasterio.features.shapes(mask):
            if value == 1:
                ground_truths.append([shape['coordinates'], 'building'])

        # Extract features from prediction
        detections = []
        for shape, value in rasterio.features.shapes(out):
            if value == 1:
                detections.append([shape['coordinates'], 1.0, 'building'])

        return detections, ground_truths

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Accumulate predictions and targets."""
        # Format
        detections, ground_truths = self._input_format(preds, target)
        # Update map computer
        self.computer.update(detections, ground_truths)

    def compute(self):
        """Compute metrics."""
        return self.computer.compute()

    def reset(self):
        """Reset metrics."""
        super().reset()
        self.computer.reset()
