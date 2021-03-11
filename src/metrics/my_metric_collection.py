"""
Implementation taken from https://github.com/PyTorchLightning/pytorch-lightning/issues/6124
"""

from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
from pytorch_lightning.metrics import Metric
from torch import nn


class MyMetricCollection(nn.ModuleDict):
    """
    MetricCollection class can be used to chain metrics that have the same
    call pattern into one single class.

    Args:
        metrics: One of the following

            * list or tuple: if metrics are passed in as a list, will use the
              metrics class name as key for output dict. Therefore, two metrics
              of the same class cannot be chained this way.

            * dict: if metrics are passed in as a dict, will use each key in the
              dict as key for output dict. Use this format if you want to chain
              together multiple of the same metric with different parameters.

    Example (input as list):

        >>> from pytorch_lightning.metrics import MetricCollection, Accuracy, Precision, Recall
        >>> target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
        >>> preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])
        >>> metrics = MetricCollection([Accuracy(),
        ...                             Precision(num_classes=3, average='macro'),
        ...                             Recall(num_classes=3, average='macro')])
        >>> metrics(preds, target)
        {'Accuracy': tensor(0.1250), 'Precision': tensor(0.0667), 'Recall': tensor(0.1111)}

    Example (input as dict):

        >>> metrics = MetricCollection({'micro_recall': Recall(num_classes=3, average='micro'),
        ...                             'macro_recall': Recall(num_classes=3, average='macro')})
        >>> metrics(preds, target)
        {'micro_recall': tensor(0.1250), 'macro_recall': tensor(0.1111)}

    """

    def __init__(
        self,
        metrics: Union[List[Metric], Tuple[Metric], Dict[str, Metric]],
        prefix=None,
    ):
        super().__init__()
        prefix = prefix if prefix is not None else ""
        if isinstance(metrics, dict):
            # Check all values are metrics
            for name, metric in metrics.items():
                if not isinstance(metric, Metric):
                    raise ValueError(
                        f"Value {metric} belonging to key {name}"
                        " is not an instance of `pl.metrics.Metric`"
                    )
                self[prefix + name] = metric
        elif isinstance(metrics, (tuple, list)):
            for metric in metrics:
                if not isinstance(metric, Metric):
                    raise ValueError(
                        f"Input {metric} to `MetricCollection` is not a instance"
                        " of `pl.metrics.Metric`"
                    )
                name = metric.__class__.__name__
                if name in self:
                    raise ValueError(f"Encountered two metrics both named {name}")
                self[prefix + name] = metric
        else:
            raise ValueError("Unknown input to MetricCollection.")

    def forward(self, *args, **kwargs) -> Dict[str, Any]:  # pylint: disable=E0202
        """
        Iteratively call forward for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        return {k: m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items()}

    def update(self, *args, **kwargs):  # pylint: disable=E0202
        """
        Iteratively call update for each metric. Positional arguments (args) will
        be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        for _, m in self.items():
            m_kwargs = m._filter_kwargs(**kwargs)
            m.update(*args, **m_kwargs)

    def compute(self) -> Dict[str, Any]:
        return {k: m.compute() for k, m in self.items()}

    def reset(self):
        """ Iteratively call reset for each metric """
        for _, m in self.items():
            m.reset()

    def clone(self):
        """ Make a copy of the metric collection """
        return deepcopy(self)

    def persistent(self, mode: bool = True):
        """Method for post-init to change if metric states should be saved to
        its state_dict
        """
        for _, m in self.items():
            m.persistent(mode)
