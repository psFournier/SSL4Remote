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
--epoch_len 12000 \
--max_epochs 500 \
--sup_batch_size 16 \
--encoder efficientnet-b0 \
--learning_rate 0.01 \
--img_aug no \
--batch_aug no \
--consistency_aug cutmix \
--num_classes 2 \
--gpus 1 \
--module mean_teacher \
--supervised_warmup 20 \
--label_decrease_factor 20 \
--ema 0.95 \
--unsup_batch_size 16 \
--crop_size 128 \
--unsup_crop_size 160 \
--datamodule miniworld_generalisation \
--cities christchurch \
--data_dir /work/OT/ai4geo/users/fournip/miniworld_tif \
--output_dir /work/OT/ai4geo/users/fournip/outputs \
--exp_name "${PBS_JOBNAME}"

module unload python/3.7.2