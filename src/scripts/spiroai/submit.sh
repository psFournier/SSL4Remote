#!/bin/bash
#SBATCH --job-name=test    # -J nom-job      => nom du job
#SBATCH --ntasks=1           # -n 24           => nombre de taches (obligatoire)
#SBATCH --time 0-2:00         # -t 0-2:00       => duree (JJ-HH:MM) (obligatoire)
#SBATCH --qos=co_long_gpu       #                 => QOS choisie (obligatoire)

ROOT=${WORKDIR}/semi-supervised-learning
PYTHON=${ROOT}/venv/bin/python
SCRIPT=${ROOT}/src/train.py
DATADIR=${WORKDIR}/data

cd $ROOT

# mpirun utilise les variables d'environnement inititalisees par Slurm
mpirun "${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule miniworld_sup \
--exp_name baseline_christchurch_profiling \
--batch_size 64 \
--data_dir "${DATADIR}"/ISPRS_VAIHINGEN \
--output_dir "${ROOT}"/outputs \
--workers 0 \
--augmentations no \
--encoder efficientnet-b0 \
--gpus 1 \
--max_epochs 5 \
--limit_train_batches 10 \
--limit_val_batches 5

