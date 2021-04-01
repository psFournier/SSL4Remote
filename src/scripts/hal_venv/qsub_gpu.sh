#!/bin/bash
#PBS -q qgpgpu
#PBS -l select=1:ncpus=4:mem=16G:ngpus=1
#PBS -l walltime=1:00:00

cd "${TMPDIR}"
LOGDIR=${TMPDIR}/outputs
DATADIR=/work/OT/ai4geo/DATA/DATASETS

"${INTERPRETER}" "${PROGRAM}" \
--exp_name "${NAME}" \
--data_dir "${DATADIR}" \
--batch_size 16 \
--unsup_loss_prop "${PARAM}" \
--nb_pass_per_epoch 100 \
--output_dir "${LOGDIR}" \
--check_val_every_n_epoch 1 \
--max_epochs 100 \
--multiple_trainloader_mode max_size_cycle \
--log_every_n_steps 10 \
--gpus 1 \
--ema 0.95 \
--nb_im_train 2 \
--nb_im_val 7 \
--nb_im_unsup_train 20 \
--encoder efficientnet-b5 \
--pretrained \
--workers 4 \
--precision 16 \
--inplaceBN \
--augmentations safe

# recopie des donnees de sortie à conserver
cp -r "${LOGDIR}" "${ROOT}"
