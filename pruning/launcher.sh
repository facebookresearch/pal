#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=modular
#SBATCH --output=/checkpoint/%u/modular/%j-%a.out
#SBATCH --error=/checkpoint/%u/modular/%j-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=%u@meta.com

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=5:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --array=1-1600


python /private/home/vivc/code/memory-theory/pruning/train.py grid --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID

# python /private/home/vivc/code/memory-theory/pruning/visualization.py get_animation --unique-id 44581400e13d49c282334b08b6060382 --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID

# python pruning/visualization.py aggregate_video -unique-id 44581400e13d49c282334b08b6060382