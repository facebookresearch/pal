#!/usr/bin/bash

python /private/home/vivc/code/memory-theory/visualization/train.py run --seed 0 --nb-epochs 10 --ffn_dim 10 --save-ext tmp --save-weights True
python /private/home/vivc/code/memory-theory/visualization/train.py run --seed 0 --nb-epochs 10 --ffn-dim 16 --save-ext tmp --save-weights True
python /private/home/vivc/code/memory-theory/visualization/train.py run --seed 0 --nb-epochs 10 --ffn-dim 16 --batch-size 20 --save-ext tmp --save-weights True

python /private/home/vivc/code/memory-theory/visualization/configs.py --save-ext tmp
python /private/home/vivc/code/memory-theory/visualization/analysis.py losses --save-ext tmp --file-format png

while read line; do
  name=$(echo $line | jq '.id')
  python /private/home/vivc/code/memory-theory/visualization/visualization.py frame --epoch 0 --file-format png --save-ext tmp --unique-id $name
done < /private/home/vivc/code/memory-theory/savings/configs/tmp.jsonl

# sbatch /private/home/vivc/code/memory-theory/launchers/launcher_bias.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/launcher_bsz.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/launcher_dim.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/launcher_dropout.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/launcher_lr.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/launcher_mlp.sh
