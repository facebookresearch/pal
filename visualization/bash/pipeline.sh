#!/usr/bin/bash

# This file showcases a typical pipeline to run the various scripts.
# Slurm launcher examples are also provided.

# train some models
python ../train.py run --seed 0 --nb-epochs 100 --ffn_dim 10 --save-ext test --save-weights True --nb-emb 3
python ../train.py run --seed 0 --nb-epochs 100 --ffn-dim 16 --save-ext test --save-weights True --nb-emb 3
python ../train.py run --seed 0 --nb-epochs 100 --ffn-dim 16 --batch-size 20 --save-ext test --save-weights True --nb-emb 3

# create an aggregated config file
python ../configs.py --save-ext test

# plots losses
python ../analysis.py losses --save-ext test --file-format png

# plots last frames
while read line; do
  name=$(echo $line | jq '.id')
  python ../visualization.py frame --epoch 100 --file-format png --save-ext test --unique-id $name
done < ../../savings/configs/test.jsonl

# create training dynamics animations
python ../visualization.py all_animation --save-ext test --num-tasks-per-videos 10 --title-key batch_size
python ../visualization.py all_aggregate --save-ext test
 
# fine-tune some models
while read line; do
  name=$(echo $line | jq '.id')
  python ../finetune.py run --save-ext test --unique-id $name --nb-epochs 100 --save_weights True
done < ../../savings/configs/test.jsonl

# create an aggregated config file
python ../configs.py --save-ext finetune/test

# plots losses
python ../analysis.py losses --save-ext finetune/test --file-format png

# # plots last frames
# while read line; do
#   name=$(echo $line | jq '.id')
#   python ../visualization.py frame --epoch 100 --file-format png --save-ext finetune/test --unique-id $name
# done < ../../savings/configs/finetune/test.jsonl

# # create training dynamics animations
# python ../visualization.py all_animation --save-ext finetune/test --num-tasks-per-videos 10 --title-key batch_size
# python ../visualization.py all_aggregate --save-ext finetune/test