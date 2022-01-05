#!/bin/bash

PYTHON=/home/pfournie/semi-supervised-learning/venv/bin/python3
SCRIPT=/home/pfournie/semi-supervised-learning/dl_toolbox/train.py

"${PYTHON}" "${SCRIPT}" \
--image_path /home/pfournie/ai4geo/data/SemcityTLS_DL/BDSD_M_3_4_7_8.tif \
--label_path /home/pfournie/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif \
--output_dir /home/pfournie/semi-supervised-learning/outputs \
--module sup \
--datamodule semcity_bdsd \
--sup_batch_size 4 \
--workers 4 \
--max_epochs 6 \
--epoch_len 5000 \
--num_classes 7 \
--img_aug color \
--batch_aug mixup \
--ignore_void \
--limit_train_batches 1 \
--limit_val_batches 1 \
--gpus 0 \
--exp_name test_semcity
