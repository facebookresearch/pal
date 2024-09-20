#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=mlp
#SBATCH --output=/checkpoint/%u/visu/mlp/%A/%a.out
#SBATCH --error=/checkpoint/%u/visu/mlp/%A/%a.err
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
#SBATCH --array=1-500

python /private/home/vivc/code/memory-theory/visualization/train.py grid --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --ablation mlp_lr --nb-seeds 100