#!/bin/bash

PYTHON=/home/pierre/PycharmProjects/RemoteSensing/venv/bin/python
SCRIPT=/home/pierre/PycharmProjects/RemoteSensing/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule airs_sup \
--exp_name test_airs \
--data_dir /home/pierre/Documents/ONERA/ai4geo/small_airs \
--output_dir /home/pierre/PycharmProjects/RemoteSensing/outputs \
--workers 0 \
--encoder efficientnet-b0 \
--max_epochs 2 \
--limit_train_batches 5 \
--limit_val_batches 2 \
--log_every_n_steps 300 \
--flush_logs_every_n_steps 1000 \
--num_sanity_val_steps 0 \
--check_val_every_n_epoch 1 \
--benchmark True