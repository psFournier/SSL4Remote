#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=01:00:00

module load python/3.7.2

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/dl_toolbox/examples/miniworld/predict.py

"${PYTHON}" "${SCRIPT}" \
--ckpt_path /work/OT/ai4geo/users/fournip/outputs/christchurch/christchurch_aug/version_0/epoch=400-step=125200.ckpt \
--gpus 1 \
--image_path /work/OT/ai4geo/users/fournip/miniworld_tif/vienna/train/0_x.tif \
--label_path /work/OT/ai4geo/users/fournip/miniworld_tif/vienna/train/0_y.tif \
--output_path /work/OT/ai4geo/users/fournip/miniworld_tif/vienna/train/0_x_pred.tif \
--workers 6 \
--batch_size 16 \
--tile_size 128 \
--tile_step 128