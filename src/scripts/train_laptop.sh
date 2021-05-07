#!/bin/bash

PYTHON=/home/pierre/PycharmProjects/RemoteSensing/venv/bin/python
SCRIPT=/home/pierre/PycharmProjects/RemoteSensing/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--exp_name test \
--data_dir /home/pierre/Documents/ONERA/ai4geo/miniworld_tif \
--output_dir /home/pierre/PycharmProjects/RemoteSensing/outputs \
--workers 0 \
--max_epochs 2 \
--limit_train_batches 5 \
--limit_val_batches 2