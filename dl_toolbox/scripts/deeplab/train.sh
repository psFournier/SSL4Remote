#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python
SCRIPT=/d/pfournie/semi-supervised-learning/dl_toolbox/train.py

"${PYTHON}" "${SCRIPT}" \
--module sup \
--datamodule mw2 \
--data_dir /scratch_ai4geo/miniworld_tif \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--workers 8 \
--max_epochs 400 \
--gpus 1 \
--city christchurch \
--exp_name test_new_code \
--train_dataset_transforms_strat hard