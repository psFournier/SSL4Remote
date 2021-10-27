#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python
SCRIPT=/d/pfournie/semi-supervised-learning/dl_toolbox/examples/train.py

"${PYTHON}" "${SCRIPT}" \
--workers 10 \
--epoch_len 2000 \
--max_epochs 50 \
--lr_milestones 25 35 45 \
--num_classes 7 \
--ignore_void \
--gpus 1 \
--module sup \
--datamodule semcity_bdsd \
--image_path /scratch_ai4geo/semcity_merged/BDSD_M_3_4_7_8.tif \
--label_path /scratch_ai4geo/semcity_merged/GT_3_4_7_8.tif \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--exp_name semcity_novoid