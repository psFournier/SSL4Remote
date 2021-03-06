#!/bin/bash
# -V
#PBS -N semisup_cpu_${NAME}
#PBS -l select=1:ncpus=1:mem=4000mb
#PBS -l walltime=01:00:00
#PBS -o out/${NAME}.out
#PBS -e err/${NAME}.err

${INTERPRETER} ${PROGRAM} --unsup_loss_prop 0.2 --nb_pass_per_epoch 1 --output_dir ${LOGDIR} --check_val_every_n_epoch 1 --max_epochs 1 --weights_summary full --multiple_trainloader_mode max_size_cycle --log_every_n_steps 10
