#!/bin/bash

# REPO=/home/pfournie/semi-supervised-learning
REPO=/d/pfournie/Documents/ai4geo/SemiSupervised/SSL4Remote
# DATA=/home/pfournie/ai4geo/data
DATA=/d/pfournie/Documents/ai4geo/data

PYTHON="${REPO}"/venv/bin/python3
SCRIPT="${REPO}"/dl_toolbox/train.py

"${PYTHON}" "${SCRIPT}" \
--image_path "${DATA}"/SemcityTLS_DL/BDSD_M_3_4_7_8.tif \
--label_path "${DATA}"/SemcityTLS_DL/GT_3_4_7_8.tif \
--output_dir "${REPO}"/outputs \
--module sup \
--datamodule semcity_bdsd \
--sup_batch_size 4 \
--workers 0 \
--max_steps 30000 \
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
