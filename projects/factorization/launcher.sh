#!/usr/bin/bash

# This file is useful to launch many experiment in parallel with slurm
# ```shell
# $ sbatch <path_to_file_folder>/launcher.sh
# ```

# Logging configuration
#SBATCH --job-name=factorization
#SBATCH --output=logs/%A/%a.out
#SBATCH --error=logs/%A/%a.err
#SBATCH --mail-type=END

# Job specification
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --array=1-500

CODE_PATH="./projects/"
python ${CODE_PATH}factorization/train.py json-grid --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID --file ${CODE_PATH}experiments/debug_experiment.json --nb-seeds 100 --save-ext debug_experiment
