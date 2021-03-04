#!/bin/bash

cd ~/Documents/ai4geo/SemiSupervised/SSL4Remote/
venv/bin/python3 src/main.py --output_dir
~/Documents/ai4geo/SemiSupervised/SSL4Remote/outputs --data_dir
~/Documents/ai4geo/data --nb_pass_per_epoch 200 --batch_size 16 --max_epochs
100 --log_every_n_steps 50 --unsup_loss_prop 0. --gpus 1
--check_val_every_n_epoch 1 --weights_summary full --multiple_trainloader_mode min_size &> outputs/output.txt

venv/bin/python3 src/main.py --output_dir
~/Documents/ai4geo/SemiSupervised/SSL4Remote/outputs --data_dir
~/Documents/ai4geo/data --nb_pass_per_epoch 200 --batch_size 16 --max_epochs
100 --log_every_n_steps 50 --unsup_loss_prop 0.2 --gpus 1
--check_val_every_n_epoch 1 --weights_summary full --multiple_trainloader_mode min_size &> outputs/output.txt

venv/bin/python3 src/main.py --output_dir
~/Documents/ai4geo/SemiSupervised/SSL4Remote/outputs --data_dir
~/Documents/ai4geo/data --nb_pass_per_epoch 200 --batch_size 16 --max_epochs
100 --log_every_n_steps 50 --unsup_loss_prop 0.5 --gpus 1
--check_val_every_n_epoch 1 --weights_summary full --multiple_trainloader_mode min_size &> outputs/output.txt