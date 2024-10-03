#!/bin/bash

# This folder should be launch from the root of the repository
# ```shell
# bash examples/bash_script.sh
# ```

# Launch some training run iteractively
python factorization/compression.py run --input-divisors '[[2, 3, 2, 5]]' --compression-rate 0.5

# to better understand the different parameters you can run the following
python factorization/compression.py run --help

# Launch experiments from json interface
# You can launch some individual runs
python factorization/compression.py json examples/config_runs.json --save-ext my_run_folder

# # You can launch a grid
python factorization/compression.py json-grid examples/config_grids.json --save-ext my_grid_folder

# You can equally overwrite some parameters from the command line
# For example, you can override all the learning rate defined in the config_grids.json
python factorization/compression.py json-grid examples/config_grids.json --save-ext my_grid_folder --learning-rate 1e1

# The results will be saved in the save directory defined from the .ini config file at the top of the repository
# To aggregate all the experiment configuration in a single json file, you can use the following command
python factorization/configs.py aggregate my_grid_folder
python factorization/configs.py aggregate my_run_folder
# You can then easily browse through the results of your runs.