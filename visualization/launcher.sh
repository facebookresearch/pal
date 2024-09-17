#!/usr/bin/bash

# In order to remove your change from git history, you can use the following command:
# ```shell
# $ git update-index --skip-worktree visualization/launcher.sh
# ```
# To track back the file, you can use the following command:
# ```shell
# $ git update-index --no-skip-worktree visualization/launcher.sh
# ```
# To list all files that are marked as skip-worktree, you can use the following command:
# ```shell
# $ git ls-files -v . | grep ^S
# ```

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


python ./train.py grid --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID

# python ./visualization.py get_animation --unique-id 44581400e13d49c282334b08b6060382 --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID

# python ./visualization.py aggregate_video -unique-id 44581400e13d49c282334b08b6060382