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
#SBATCH --array=1-50

# TODO: make it nicer
python ./visualization.py get_animation --unique-id 44581400e13d49c282334b08b6060382 --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID
python ./visualization.py aggregate_video -unique-id 44581400e13d49c282334b08b6060382