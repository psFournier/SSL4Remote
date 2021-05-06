#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=8:mem=92G:ngpus=1
#PBS -l walltime=1:00:00

cd "${TMPDIR}"
mkdir miniworld
cp -r /work/OT/ai4geo/users/fournip/miniworld_tif/austin miniworld_tif/

PYTHON=/home/eh/fournip/SemiSupervised/SSL4Remote/venv/bin/python
SCRIPT=/home/eh/fournip/SemiSupervised/SSL4Remote/src/train.py

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule airs_sup \
--exp_name test_airs \
--data_dir "${TMPDIR}"/miniworld_tif \
--output_dir ${TMPDIR}/outputs \
--workers 0 \
--encoder efficientnet-b0 \
--max_epochs 2 \
--limit_train_batches 5 \
--limit_val_batches 2 \
--log_every_n_steps 300 \
--flush_logs_every_n_steps 1000 \
--num_sanity_val_steps 0 \
--check_val_every_n_epoch 1 \
--benchmark True \
--gpus 1

# recopie des donnees de sortie à conserver
cp -r ${TMPDIR}/outputs /home/eh/fournip/SemiSupervised/SSL4Remote