#!/bin/bash

"${ROOT}"/venv/bin/python "${PROGRAM}" \
--exp_name "${NAME}" \
--data_dir "${DATADIR}" \
--batch_size 10 \
--unsup_loss_prop 1 \
--nb_pass_per_epoch 10 \
--output_dir "${LOGDIR}" \
--check_val_every_n_epoch 1 \
--max_epochs 3 \
--multiple_trainloader_mode min_size \
--log_every_n_steps 10 \
--ema 0.95 \
--nb_im_train 2 \
--nb_im_val 7 \
--nb_im_unsup_train 20 \
--encoder efficientnet-b5 \
--workers 4 \
--precision 32 \
--augmentations safe
