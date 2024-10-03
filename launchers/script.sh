#!/bin/bash

CODE_PATH="/private/home/vivc/code/memory-theory"

# slurm launcher
sbatch ${CODE_PATH}/launchers/compression.sh

# python ${CODE_PATH}/factorization/configs.py aggregate debug_experiment