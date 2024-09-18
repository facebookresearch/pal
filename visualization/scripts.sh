#!/usr/bin/bash

# python /private/home/vivc/code/memory-theory/visualization/configs.py --save-ext tmp
python /private/home/vivc/code/memory-theory/visualization/analysis.py losses --save-ext tmp --file-format png

# python /private/home/vivc/code/memory-theory/visualization/analysis.py losses --file-format png
# python /private/home/vivc/code/memory-theory/visualization/analysis.py frame --unique-id 2721c52b4f004998a6c55d093cfeec93 --epoch 900 --file-format png

# python /private/home/vivc/code/memory-theory/visualization/train.py run --seed 0 --nb-epochs 10 --ffn_dim 10 --save-ext tmp --save-weights True
# python /private/home/vivc/code/memory-theory/visualization/train.py run --seed 0 --nb-epochs 10 --ffn-dim 16 --save-ext tmp --save-weights True
# python /private/home/vivc/code/memory-theory/visualization/train.py run --seed 0 --nb-epochs 10 --ffn-dim 16 --batch-size 20 --save-ext tmp --save-weights True
