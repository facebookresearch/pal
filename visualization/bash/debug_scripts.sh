#!/usr/bin/bash

python ../train.py run --seed 0 --nb-epochs 10 --ffn_dim 10 --save-ext tmp --save-weights True
python ../train.py run --seed 0 --nb-epochs 10 --ffn-dim 16 --save-ext tmp --save-weights True
python ../train.py run --seed 0 --nb-epochs 10 --ffn-dim 16 --batch-size 20 --save-ext tmp --save-weights True
python ../configs.py --save-ext tmp
python ../analysis.py losses --save-ext tmp --file-format png

while read line; do
  name=$(echo $line | jq '.id')
  python ../visualization.py frame --epoch 0 --file-format png --save-ext tmp --unique-id $name
  python ../visualization.py animation --save-ext tmp --unique-id $name
done < ../../savings/configs/tmp.jsonl
