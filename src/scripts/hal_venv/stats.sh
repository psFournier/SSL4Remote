#!/bin/bash

module load python/3.7.2

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/src/stat_scores_image.py

"${PYTHON}" "${SCRIPT}" \
--label_path /work/OT/ai4geo/users/fournip/miniworld_tif/vienna/train/0_y.tif \
--pred_path /work/OT/ai4geo/users/fournip/outputs/sup_mw_austin_31-5/version_0/checkpoints/vienna_train_0.tif \
--output_path /work/OT/ai4geo/users/fournip/outputs/sup_mw_austin_31-5/version_0/checkpoints/vienna_train_0_stats0.tif \
--class_id 1
