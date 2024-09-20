
# sbatch /private/home/vivc/code/memory-theory/launchers/train_bsz.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_dim.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_lr.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_mlp.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/train_seed.sh

# python /private/home/vivc/code/memory-theory/visualization/configs.py --save-ext batch_size
# python /private/home/vivc/code/memory-theory/visualization/configs.py --save-ext ffn_dim
# python /private/home/vivc/code/memory-theory/visualization/configs.py --save-ext lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py --save-ext mlp_lr
# python /private/home/vivc/code/memory-theory/visualization/configs.py --save-ext seed

# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --seed True
# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --seed False --key test_acc
# python /private/home/vivc/code/memory-theory/visualization/analysis.py ablation --seed False --key success

# sbatch /private/home/vivc/code/memory-theory/launchers/visu_bsz.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/visu_dim.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/visu_lr.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/visu_mlp.sh
# sbatch /private/home/vivc/code/memory-theory/launchers/visu_seed.sh

# python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext batch_size
# python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext ffn_dim
# python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext lr
# python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext mlp_lr
# python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext seed

