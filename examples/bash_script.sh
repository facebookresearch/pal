#!/bin/bash

# Change this variable to put your own path
CODE_PATH="/private/home/vivc/code/memory-theory"

# This folder can be launch from bash, for example from the root of the repository
# ```shell
# bash examples/bash_script.sh
# ```

# Launch some training run iteractively
python ${CODE_PATH}/factorization/compression.py run --log-input-factors '[2, 2, 3]'

# to better understand the different parameters you can run the following
python ${CODE_PATH}/factorization/compression.py run --help

# Launch experiments from json interface
# You can launch some individual runs
python ${CODE_PATH}/factorization/compression.py json ${CODE_PATH}/examples/config_runs.json --save-ext my_run_folder

# You can launch a grid
# python factorization/compression.py json-grid examples/config_grids.json --save-ext my_grid_folder

# You can equally overwrite some parameters from the command line
# For example, you can override all the learning rate defined in the config_grids.json
# python factorization/compression.py json-grid examples/config_grids.json --save-ext my_grid_folder --learning-rate 1e1

# The results will be saved in the save directory defined from the .ini config file at the top of the repository
# To aggregate all the experiment configuration in a single json file, you can use the following command
# python factorization/configs.py aggregate my_grid_folder
python ${CODE_PATH}/factorization/configs.py aggregate my_run_folder
# You can then easily browse through the results of your runs.