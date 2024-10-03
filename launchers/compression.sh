#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=compression
#SBATCH --output=/checkpoint/%u/factorization/%A/%a.out
#SBATCH --error=/checkpoint/%u/factorization/%A/%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=%u@meta.com

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --array=1-500

CODE_PATH="/private/home/vivc/code/memory-theory"
python ${CODE_PATH}/factorization/compression.py json-grid --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --file ${CODE_PATH}/experiments/debug_experiment.json --nb-seeds 100
