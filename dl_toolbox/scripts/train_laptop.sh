#!/bin/bash

PYTHON=/home/pierre/PycharmProjects/RemoteSensing/venv/bin/python
SCRIPT=/home/pierre/PycharmProjects/RemoteSensing/dl_toolbox/examples/miniworld/train.py

"${PYTHON}" "${SCRIPT}" \
--module sup \
--datamodule mw \
--data_dir /home/pierre/Documents/ONERA/ai4geo/miniworld_tif \
--output_dir /home/pierre/PycharmProjects/RemoteSensing/outputs \
--workers 8 \
--max_epochs 500 \
--city austin \
--train_val 2 5 \
--tta_augment hsv contrast