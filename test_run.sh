#!/bin/bash

#SBATCH --job-name=cpDL
#SBATCH --time=7-24:00:00
#SBATCH --error=/home/zc95/error/error8.txt
#SBATCH --mem=20GB
#SBATCH --array=1-6
#SBATCH -c12
set -o errexit

module load anaconda3/2019.07
conda activate zyai
srun python test_run.py $SLURM_ARRAY_TASK_ID
