#!/usr/bin/bash
#SBATCH --job-name=wsi_im
#SBATCH --output=wsi_im.out
#SBATCH --error=wsi_im.err
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