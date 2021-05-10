#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python
SCRIPT=/d/pfournie/semi-supervised-learning/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--data_dir /scratch_ai4geo/miniworld_tif \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--workers 12 \
--max_epochs 500 \
--gpus 1 \
--city austin \
--train_val 2 5 \
--tta_augment hsv contrast