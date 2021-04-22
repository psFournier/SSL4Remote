#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=16G:ngpus=1
#PBS -l walltime=1:00:00

ROOT=/home/eh/fournip/SemiSupervised/SSL4Remote
PYTHON=${ROOT}/venv/bin/python
SCRIPT=${ROOT}/src/train.py

cd "${TMPDIR}"
cp -r /work/OT/ai4geo/DATA_NEW/GROUNDTRUTH/ISPRS_VAIHINGEN .
LOGDIR=${TMPDIR}/outputs

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--exp_name baseline_christchurch_profiling \
--batch_size 64 \
--data_dir "${TMPDIR}"/ISPRS_VAIHINGEN \
--output_dir "${LOGDIR}" \
--workers 0 \
--augmentations no \
--encoder efficientnet-b0 \
--gpus 1 \
--max_epochs 5 \
--limit_train_batches 10 \
--limit_val_batches 5

# recopie des donnees de sortie à conserver
cp -r "${LOGDIR}" "${ROOT}"
