#!/bin/bash

#SBATCH --job-name=cpDL
#SBATCH --time=7-24:00:00
#SBATCH --output=/home/zc95/output/output_%a.txt/
#SBATCH --error=/home/zc95/error/error_%a.txt
#SBATCH --mem=20GB
#SBATCH --array=1,3,7,9,13,15,19,21,25,27,31,33
#SBATCH -c12
set -o errexit

source activate fastai
srun python test_run.py $SLURM_ARRAY_TASK_ID
