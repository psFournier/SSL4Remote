#!/bin/bash
#SBATCH --job-name=test    # -J nom-job      => nom du job
#SBATCH --ntasks=1           # -n 24           => nombre de taches (obligatoire)
#SBATCH --time 0-2:00         # -t 0-2:00       => duree (JJ-HH:MM) (obligatoire)
#SBATCH --qos=co_long_gpu       #                 => QOS choisie (obligatoire)

PYTHON=${WORKDIR}/semi-supervised-learning/venv/bin/python
SCRIPT=${WORKDIR}/semi-supervised-learning/dlcooker_pfournie/train.py

cd ${WORKDIR}/semi-supervised-learning

"${PYTHON}" "${SCRIPT}" \
--module supervised_baseline \
--datamodule airs_sup \
--exp_name test_airs \
--data_dir ${WORKDIR}/data \
--output_dir ${WORKDIR}/semi-supervised-learning/outputs \
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