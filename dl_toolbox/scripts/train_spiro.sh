#!/bin/bash
#SBATCH --job-name=test    # -J nom-job      => nom du job
#SBATCH --ntasks=1           # -n 24           => nombre de taches (obligatoire)
#SBATCH --time 0-2:00         # -t 0-2:00       => duree (JJ-HH:MM) (obligatoire)
#SBATCH --qos=co_long_gpu       #                 => QOS choisie (obligatoire)
echo "not implemented"
