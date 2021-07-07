#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python
SCRIPT=/d/pfournie/semi-supervised-learning/dlcooker_pfournie/train.py

"${PYTHON}" "${SCRIPT}" \
--module sup \
--datamodule mw \
--data_dir /scratch_ai4geo/miniworld_tif \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--workers 12 \
--max_epochs 5 \
--gpus 1 \
--city austin \
--train_val 2 5 \
--tta hsv contrast \
--img_augment d4 hue \
--batch_augment cutmix