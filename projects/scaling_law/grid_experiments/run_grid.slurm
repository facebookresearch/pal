#!/usr/bin/bash

#SBATCH --time 01:00:00
#SBATCH -p ccm
#SBATCH -c 1
#SBATCH --output=log/out-assoc-%j.log
#SBATCH --error=log/err-assoc-%j.log
#SBATCH --array=1-500

export OMP_NUM_THREADS=1

source ~/.bashrc
source $VENVDIR/fillm/bin/activate

python run.py --num_tasks $SLURM_ARRAY_TASK_COUNT --task_id $SLURM_ARRAY_TASK_ID --name finer_dim
