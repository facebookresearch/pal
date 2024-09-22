#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=video-agg
#SBATCH --output=/checkpoint/%u/video/%A/%a.out
#SBATCH --error=/checkpoint/%u/video/%A/%a.err

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=0

python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext seed
python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext finetune