#!/bin/bash

PYTHON=/d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/venv/bin/python
SCRIPT=/d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/dl_toolbox/examples/train.py

"${PYTHON}" "${SCRIPT}" \
--workers 0 \
--epoch_len 5000 \
--max_steps 10 \
--sup_batch_size 16 \
--encoder efficientnet-b0 \
--learning_rate 0.01 \
--img_aug no \
--batch_aug no \
--consistency_aug cutmix \
--num_classes 2 \
--gpus 1 \
--module mean_teacher \
--supervised_warmup 0 \
--label_decrease_factor 20 \
--ema 0.95 \
--unsup_batch_size 16 \
--crop_size 128 \
--unsup_crop_size 160 \
--datamodule miniworld_generalisation \
--cities christchurch \
--data_dir /d/pfournie/Documents/ai4geo/data/miniworld_tif \
--output_dir /d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote/outputs \
--exp_name airs \
--limit_train_batches 2 \
--limit_val_batches 2