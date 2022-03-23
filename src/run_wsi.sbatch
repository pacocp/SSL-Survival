#!/usr/bin/bash
#SBATCH --job-name=wsi_dir
#SBATCH --output=wsi_dir.out
#SBATCH --error=wsi_dir.err
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4

# load the module
ml load python/3.9.0

# activate virtualenv
source ~/code/wsi_mutation/bin/activate
 
# run code
bash ~/code/SSL-Survival/src/run_wsi.bash