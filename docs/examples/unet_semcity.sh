#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python3
SCRIPT=/d/pfournie/semi-supervised-learning/docs/examples/unet_semcity.py

"${PYTHON}" "${SCRIPT}" \
--data_path /scratchf/semcity_merged \
--splitfile_path /d/pfournie/split_semcity.csv \
--test_fold 4 \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--sup_batch_size 8 \
--workers 6 \
--max_epochs 10 \
--epoch_len 2000 \
--crop_size 256 \
--num_classes 8 \
--img_aug d4_color-0 \
--batch_aug no \
--encoder efficientnet-b5 \
--in_channels 3 \
--initial_lr 0.05 \
--final_lr 0.001 \
--lr_milestones 0.5 0.9 \
--weight_decay 0. \
--pretrained \
--train_with_void \
--limit_train_batch 1 \
--limit_val_batch 1 \
--gpus 1
