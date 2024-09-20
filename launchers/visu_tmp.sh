#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=video-seed
#SBATCH --output=/checkpoint/%u/video/%j-%a.out
#SBATCH --error=/checkpoint/%u/video/%j-%a.err

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=0
#SBATCH --array=1-100

python /private/home/vivc/code/memory-theory/visualization/visualization.py all_animation --save-ext seed --num-tasks $SLURM_ARRAY_TASK_COUNT --num-tasks-per-videos 20 --task-id $SLURM_ARRAY_TASK_ID --title-key seed
