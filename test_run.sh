#!/bin/bash
#SBATCH --time=08:00:00 
#SBATCH --mem=2GB
#SBATCH --array=1-6
set -o errexit
module load anaconda3/2019.07
python3 test_run.py $SLURM_ARRAY_TASK_ID
