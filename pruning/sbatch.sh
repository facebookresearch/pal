#!/bin/zsh

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=scavenge
#SBATCH --array=1-500
#SBATCH --signal=B:SIGUSR1@120

#SBATCH --job-name=exp
#SBATCH --output=log/%j/%j.out
#SBATCH --error=log/%j/%j.err
#SBATCH --open-mode=append

ARRAY_SIZE=500
srun python ./launcher.py --array_size $ARRAY_SIZE --task $SLURM_ARRAY_TASK_ID