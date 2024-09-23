
# # Launch Ablation Studies
# sbatch /private/home/vivc/code/memory-theory/launchers/train_bsz.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_dim.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_lr.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_mlp.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_seed.sh

# # Aggreagte Configurations
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext batch_size
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext ffn_dim
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext mlp_lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext seed

# # Filter Successful Configurations
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext batch_size
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext ffn_dim
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext mlp_lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext seed

# # Launch Finetuning Ablation Study
# sbatch /private/home/vivc/code/memory-theory/launchers/finetune_seed.sh

# # Aggregrate and Filter Finetuning Config
# python /private/home/vivc/code/memory-theory/visualization/configs.py aggregate --save-ext finetune/seed
# python /private/home/vivc/code/memory-theory/visualization/configs.py filter --save-ext finetune/seed

# # Launch Animation Generations
# sbatch /private/home/vivc/code/memory-theory/launchers/visu_seed.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/visu_finetune.sh

# # Aggregate Animations
# sbatch /private/home/vivc/code/memory-theory/launchers/aggregate.sh

# # Ablation study plots
# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --seed
# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --noseed --key test_acc
# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --noseed --key success

# # Plots First and Last Frames of Successful Finetuning
# while read line; do
#   name=$(echo $line | jq '.id')
#   python /private/home/vivc/code/memory-theory/visualization/visualization.py frame --epoch 0 --file-format png --save-ext finetune/seed --suffix success --unique-id $name
#   python /private/home/vivc/code/memory-theory/visualization/visualization.py frame --epoch 1000 --file-format png --save-ext finetune/seed --suffix success --unique-id $name
# done < /private/home/vivc/code/memory-theory/configs/success/finetune/seed.jsonl

# while read line; do
#   name=$(echo $line | jq '.id')
#   python /private/home/vivc/code/memory-theory/visualization/visualization.py frame --epoch 0 --file-format png --save-ext seed --suffix success --unique-id $name
#   python /private/home/vivc/code/memory-theory/visualization/visualization.py frame --epoch 1000 --file-format png --save-ext seed --suffix success --unique-id $name
# done < /private/home/vivc/code/memory-theory/configs/success/seed.jsonl