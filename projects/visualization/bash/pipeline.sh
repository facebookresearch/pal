#!/usr/bin/bash

# This file showcases a typical pipeline to run the various scripts.
# Slurm launcher examples are also provided.

mv ../../user_config.ini ../../.user_config.ini
cat > ../../user_config.ini <<EOF
[Relative Paths]
config_dir = test/configs
image_dir = test/images
save_dir = test/results
EOF

# train some models
python ../train.py run --seed 0 --nb-epochs 100 --ffn_dim 10 --save-ext test --save-weights True --nb-emb 3
python ../train.py run --seed 0 --nb-epochs 100 --ffn-dim 16 --save-ext test --save-weights True --nb-emb 3
python ../train.py run --seed 0 --nb-epochs 100 --ffn-dim 16 --batch-size 20 --save-ext test --save-weights True --nb-emb 3

# create an aggregated config file
python ../configs.py aggregate --save-ext test

# plots losses
python ../analysis.py losses --save-ext test --file-format png

# plots last frames
while read line; do
  name=$(echo $line | jq '.id')
  python ../visualization.py frame --epoch 100 --file-format png --save-ext test --unique-id $name
done < ../../test/configs/test.jsonl

# create training dynamics animations
python ../visualization.py all_animation --save-ext test --num-tasks-per-videos 10 --title-key batch_size
python ../visualization.py all_aggregate --save-ext test
 
# fine-tune some models
while read line; do
  name=$(echo $line | jq '.id')
  python ../finetune.py run --save-ext test --unique-id $name --nb-epochs 100 --save_weights True
done < ../../test/configs/test.jsonl

# create an aggregated config file
python ../configs.py aggregate --save-ext finetune/test

# plots losses
python ../analysis.py losses --save-ext finetune/test --file-format png

# plots last frames
while read line; do
  name=$(echo $line | jq '.id')
  python ../visualization.py frame --epoch 100 --file-format png --save-ext finetune/test --unique-id $name
done < ../../test/configs/finetune/test.jsonl

# create training dynamics animations
python ../visualization.py all_animation --save-ext finetune/test --num-tasks-per-videos 10 --title-key batch_size
python ../visualization.py all_aggregate --save-ext finetune/test

rm -rf ../../test/
rm ../../user_config.ini
mv ../../.user_config.ini ../../user_config.ini