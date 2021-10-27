#!/bin/bash

PYTHON=/d/pfournie/semi-supervised-learning/venv/bin/python
SCRIPT=/d/pfournie/semi-supervised-learning/dl_toolbox/examples/train.py

"${PYTHON}" "${SCRIPT}" \
--workers 10 \
--max_epochs 150 \
--lr_milestones 75 100 125 \
--num_classes 2 \
--gpus 1 \
--module sup \
--datamodule miniworld_generalisation \
--cities christchurch \
--data_dir /scratch_ai4geo/miniworld_tif \
--output_dir /d/pfournie/semi-supervised-learning/outputs \
--exp_name airs