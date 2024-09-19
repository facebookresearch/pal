#!/usr/bin/bash

# This file showcases a typical pipeline to run the various scripts.
# Train is launch manually, it could be launched with a slurm launcher as well, c.f. `train_launcher.sh`.

python ../train.py run --seed 0 --ffn_dim 10 --save-ext tmp --save-weights True
python ../train.py run --seed 0 10 --ffn-dim 16 --save-ext tmp --save-weights True
python ../train.py run --seed 0 10 --ffn-dim 16 --batch-size 20 --save-ext tmp --save-weights True
python ../configs.py --save-ext tmp
python ../analysis.py losses --save-ext tmp --file-format png

while read line; do
  name=$(echo $line | jq '.id')
  python ../visualization.py frame --epoch 0 --file-format png --save-ext tmp --unique-id $name
done < ../../savings/configs/tmp.jsonl

sbatch ./visu_launcher.sh
# python ./visualization.py all_aggregate --save-ext tmp --num-tasks $SLURM_ARRAY_TASK_COUNT --task-id $SLURM_ARRAY_TASK_ID
