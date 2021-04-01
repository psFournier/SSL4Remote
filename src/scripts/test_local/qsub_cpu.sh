#!/bin/bash

LOGDIR=${ROOT}/outputs

${INTERPRETER} ${PROGRAM} --unsup_loss_prop ${PARAM} --nb_pass_per_epoch 10 --output_dir ${LOGDIR} --check_val_every_n_epoch 1 --max_epochs 2 --multiple_trainloader_mode max_size_cycle --log_every_n_steps 10 --ema 0.95
