#!/usr/bin/bash
#SBATCH --job-name=wsi_job
#SBATCH --output=wsi_job.out
#SBATCH --error=wsi_job.err
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --cpus-per-gpu=10

# load the module
ml load python/3.9.0

# activate virtualenv
source ~/code/wsi_mutation/bin/activate
 
# run code
bash ~/code/SSL-Survival/src/run_wsi.bash