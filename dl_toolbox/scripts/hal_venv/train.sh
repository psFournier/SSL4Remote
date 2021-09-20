#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=6:mem=92G:ngpus=1
#PBS -l walltime=12:00:00

module load python/3.7.2
cd "${TMPDIR}"
#mkdir miniworld_tif
#CITY=austin
#cp -r /work/OT/ai4geo/users/fournip/miniworld_tif/"${CITY}" miniworld_tif/

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venvpython37/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/dl_toolbox/examples/semcity/train.py

"${PYTHON}" "${SCRIPT}" \
--output_dir /work/OT/ai4geo/users/fournip/outputs \
--workers 6 \
--max_epochs 100 \
--num_classes 8 \
--gpus 1 \
--module sup \
--datamodule bdsd \
--image_path /work/OT/ai4geo/users/fournip/semcity_merged/BDSD_M_3_4_7_8.tif \
--label_path /work/OT/ai4geo/users/fournip/semcity_merged/GT_3_4_7_8.tif \
--exp_name "${PBS_JOBNAME}"

module unload python/3.7.2