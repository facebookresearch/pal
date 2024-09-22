
# sbatch /private/home/vivc/code/memory-theory/launchers/train_bsz.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_dim.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_lr.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_mlp.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_seed.sh

# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext batch_size
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext ffn_dim
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext mlp_lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext seed

# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext batch_size
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext ffn_dim
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext mlp_lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext seed

# sbatch /private/home/vivc/code/memory-theory/launchers/finetune_seed.sh

# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext finetune/seed
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext finetune/seed

# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --seed
# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --noseed --key test_acc
# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --noseed --key success

# sbatch /private/home/vivc/code/memory-theory/launchers/visu_seed.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/visu_finetune.sh

# sbatch /private/home/vivc/code/memory-theory/launchers/aggregate.sh

# # plots first and last frames of successful finetuning
# while read line; do
#   name=$(echo $line | jq '.id')
#   python /private/home/vivc/code/memory-theory/visualization/visualization.py frame --epoch 0 --file-format png --save-ext finetune/seed --suffix success --unique-id $name
#   python /private/home/vivc/code/memory-theory/visualization/visualization.py frame --epoch 1000 --file-format png --save-ext finetune/seed --suffix success --unique-id $name
# done < /private/home/vivc/code/memory-theory/configs/success/finetune/seed.jsonl
