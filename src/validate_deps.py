#!/usr/bin/python3

try:
    import pytorch_lightning as pl

    print("pytorch_lightning PASS %s" % (pl.__version__))
except:
    print("pytorch_lightning FAIL")

try:
    import map_metric_api

    print("map_metric_api PASS")
except:
    print("map_metric_api FAIL")

try:
    import tqdm

    print("tqdm PASS %s" % (tqdm.__version__))
except:
    print("tqdm FAIL")

try:
    import geojson

    print("geojson PASS %s" % (geojson.__version__))
except:
    print("geojson FAIL")

try:
    import mercantile

    print("mercantile PASS %s" % (mercantile.__version__))
except:
    print("mercantile FAIL")


try:
    import matplotlib

    print("matplotlib PASS %s" % (matplotlib.__version__))
except:
    print("matplotlib FAIL")

try:
    import numpy

    print("numpy PASS %s" % (numpy.__version__))
except:
    print("numpy FAIL")

try:
    import scipy

    print("scipy PASS %s" % (scipy.__version__))
except:
    print("scipy FAIL")

try:  # pip install pytorch-lightning-bolts
    import pl_bolts

    print("pl_bolts PASS %s" % (pl_bolts.__version__))
except:
    print("pl_bolts FAIL")

try:
    import torchvision

    print("torchvision PASS %s" % (torchvision.__version__))
except:
    print("torchvision FAIL")

try:
    import PIL

    print("PIL PASS %s" % (PIL.__version__))
except:
    print("PIL FAIL")

try:
    import rasterio

    print("rasterio PASS %s" % (rasterio.__version__))
except:
    print("rasterio FAIL")

try:
    import hydra

    print("hydra PASS %s" % (hydra.__version__))
except:
    print("hydra FAIL")

try:
    import albumentations

    print("albumentations PASS %s" % (albumentations.__version__))
except:
    print("albumentations FAIL")


try:
    import seaborn

    print("seaborn PASS %s" % (seaborn.__version__))
except:
    print("seaborn FAIL")


try:
    import segmentation_models_pytorch as smp

    print("segmentation_models_pytorch PASS %s" % (smp.__version__))
except:
    print("segmentation_models_pytorch FAIL")

try:
    import torch

    print("torch PASS %s" % (torch.__version__))
except:
    print("torch FAIL")
