#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python
SCRIPT=/d/pfournie/semi-supervised-learning/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--exp_name austin \
--data_dir /scratch_ai4geo/miniworld_tif \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--workers 20 \
--augmentations no \
--encoder efficientnet-b0 \
--max_epochs 2 \
--limit_train_batches 5 \
--limit_val_batches 2 \
--log_every_n_steps 300 \
--flush_logs_every_n_steps 1000 \
--num_sanity_val_steps 0 \
--check_val_every_n_epoch 1 \
--benchmark True \
--gpus 1 \
--city austin