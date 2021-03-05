#!/bin/bash
for i in $(seq 0.0 0.2 1.0)
do
   venv/bin/python3 src/main.py --output_dir ~/Documents/ai4geo/SemiSupervised/SSL4Remote/outputs --data_dir ~/Documents/ai4geo/data --nb_pass_per_epoch 10 --batch_size 16 --max_epochs 2 --log_every_n_steps 50 --unsup_loss_prop "$i" --gpus 1 --check_val_every_n_epoch 1 --weights_summary full --multiple_trainloader_mode min_size &> outputs/output.txt
done