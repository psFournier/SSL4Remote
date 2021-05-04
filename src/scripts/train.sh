#!/bin/bash

PYTHON=${ROOTDIR}/venv/bin/python
SCRIPT=${ROOTDIR}/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule airs_sup \
--exp_name test_airs \
--data_dir "${DATADIR}"/small_airs \
--output_dir "${LOGDIR}" \
--workers 0 \
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
--gpus 1