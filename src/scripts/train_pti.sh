#!/bin/bash

PYTHON=/d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/venv/bin/python
SCRIPT=/d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--data_dir /d/pfournie/Documents/ai4geo/data/miniworld_tif \
--output_dir /d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/outputs \
--workers 8 \
--max_epochs 500 \
--gpus 1 \
--city austin \
--train_val 2 5 \
--tta_augment hsv contrast