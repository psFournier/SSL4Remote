#!/bin/bash

HOME=/d/pfournie/semi-supervised-learning
PYTHON="$HOME"/venv/bin/python
SCRIPT="$HOME"/src/test.py

"${PYTHON}" "${SCRIPT}" \
--ckpt_path "$HOME"/outputs/sup_mw_austin_2-2_d4/version_1/checkpoints/epoch=0-step=4.ckpt \
--data_dir /scratch_ai4geo/miniworld_tif \
--gpus 1