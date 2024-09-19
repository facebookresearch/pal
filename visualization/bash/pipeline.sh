#!/usr/bin/bash

# This file showcases a typical pipeline to run the various scripts.
# Slurm launcher examples are also provided.

python ../train.py run --seed 0 --nb-epochs 100 --ffn_dim 10 --save-ext tmp --save-weights True
python ../train.py run --seed 0 --nb-epochs 100 --ffn-dim 16 --save-ext tmp --save-weights True
python ../train.py run --seed 0 --nb-epochs 100 --ffn-dim 16 --batch-size 20 --save-ext tmp --save-weights True
python ../configs.py --save-ext tmp
python ../analysis.py losses --save-ext tmp --file-format png

while read line; do
  name=$(echo $line | jq '.id')
  python ../visualization.py frame --epoch 0 --file-format png --save-ext tmp --unique-id $name
done < ../../savings/configs/tmp.jsonl


python ../visualization.py all_animation --save-ext tmp --num-tasks-per-videos 10 --title-key batch_size
python ../visualization.py all_aggregate --save-ext tmp
