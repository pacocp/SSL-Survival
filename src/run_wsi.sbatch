#!/usr/bin/bash
#SBATCH --job-name=wsi_dir150
#SBATCH --output=wsi_dir150.out
#SBATCH --error=wsi_dir150.err
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=10 
#SBATCH -C GPU_MEM:32GB

# load the module #SBATCH -C GPU_MEM:32GB
ml load python/3.9.0

# activate virtualenv
source ~/code/wsi_mutation/bin/activate
 
# run code
# bash ~/code/SSL-Survival/src/run_wsi.bash
python  ~/code/SSL-Survival/src/wsi_model.py --config ~/code/SSL-Survival/configs/config_WSI.json \
                                                --save_dir ~/code/SSL-Survival/checkpoints/wsi_training/wsi_dir150/ \
                                                --num_samples 150 \
                                                --seeds 86,54 \
                                                #  0,9,42,2022,61,72,99,273