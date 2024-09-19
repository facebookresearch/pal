#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=visu
#SBATCH --output=~/%u/visu/%j-%a.out
#SBATCH --error=~/%u/visu/%j-%a.err

# Job specification
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --array=1-90

# TODO: make it nicer
python ./visualization.py all_animation --save-ext tmp --num-tasks $SLURM_ARRAY_TASK_COUNT --num-tasks-per-videos 30 --task-id $SLURM_ARRAY_TASK_ID --title-key batch_size
