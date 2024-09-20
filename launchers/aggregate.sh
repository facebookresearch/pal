
#!/usr/bin/bash

# Logging configuration
#SBATCH --job-name=video-seed
#SBATCH --output=/checkpoint/%u/video/%j-%a.out
#SBATCH --error=/checkpoint/%u/video/%j-%a.err

# Job specification
#SBATCH --partition=scavenge
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=0

python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext batch_size
python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext ffn_dim
python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext lr
python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext mlp_lr
python /private/home/vivc/code/memory-theory/visualization/visualization.py all_aggregate --save-ext seed