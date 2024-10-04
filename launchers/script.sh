#!/bin/bash

CODE_PATH="./"

# slurm launcher
sbatch ${CODE_PATH}launchers/compression.sh

# python ${CODE_PATH}factorization/configs.py aggregate debug_experiment