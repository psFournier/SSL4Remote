#!/bin/bash
#PBS -q qgpgpudev
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python/3.7.2
cd "${TMPDIR}"

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/dl_toolbox/examples/train.py

"${PYTHON}" "${SCRIPT}" \
--workers 6 \
--epoch_len 2000 \
--max_epochs 50 \
--encoder efficientnet-b0 \
--learning_rate 0.05 \
--img_aug d4 \
--batch_aug no \
--num_classes 7 \
--ignore_void \
--gpus 1 \
--module sup \
--datamodule semcity_bdsd \
--image_path /work/OT/ai4geo/users/fournip/semcity_merged/BDSD_M_3_4_7_8.tif \
--label_path /work/OT/ai4geo/users/fournip/semcity_merged/GT_3_4_7_8.tif \
--output_dir /work/OT/ai4geo/users/fournip/outputs \
--exp_name "${PBS_JOBNAME}"

module unload python/3.7.2