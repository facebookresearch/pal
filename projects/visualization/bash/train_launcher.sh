#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=visu
#SBATCH --output=./logs/visu/%A/%a.out
#SBATCH --error=./logs/visu/%A/%a.err

# Job specification
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --array=1-50

python ./train.py grid --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID
