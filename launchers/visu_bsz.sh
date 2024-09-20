#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=video-bsz
#SBATCH --output=/checkpoint/%u/video/%A/%a.out
#SBATCH --error=/checkpoint/%u/video/%A/%a.err

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=0
#SBATCH --array=1-500

python /private/home/vivc/code/memory-theory/visualization/visualization.py all_animation --save-ext batch_size --num-tasks $SLURM_ARRAY_TASK_COUNT --num-tasks-per-videos 20 --task-id $SLURM_ARRAY_TASK_ID --title-key batch_size