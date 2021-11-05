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
--epoch_len 5000 \
--max_steps 100000 \
--sup_batch_size 16 \
--encoder efficientnet-b0 \
--learning_rate 0.01 \
--img_aug d4_color \
--batch_aug cutmix \
--consistency_aug color \
--num_classes 7 \
--ignore_void \
--gpus 1 \
--module mean_teacher \
--supervised_warmup 0 \
--ema 0.95 \
--unsup_batch_size 16 \
--crop_size 128 \
--unsup_crop_size 160 \
--consistency_training \
--pseudo_labelling \
--do_semisup \
--datamodule semcity_bdsd \
--image_path /work/OT/ai4geo/users/fournip/semcity_merged/BDSD_M_3_4_7_8.tif \
--label_path /work/OT/ai4geo/users/fournip/semcity_merged/GT_3_4_7_8.tif \
--data_dir /work/OT/ai4geo/users/fournip/semcity_merged/test \
--output_dir /work/OT/ai4geo/users/fournip/outputs/semcity/sup \
--exp_name "${PBS_JOBNAME}"
#--do_semisup \

module unload python/3.7.2