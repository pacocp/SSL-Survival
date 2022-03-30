foldername=ssl_10 #$(date +%Y-%m-%d-%H-%M-%S)
sourcefolder=/home/users/mpizuric/code/SSL-Survival
destfolder=$sourcefolder/checkpoints/wsi_training/$foldername/
mkdir $destfolder
codefile=wsi_model.py 
python $codefile --config $sourcefolder/configs/config_WSI.json \
                    --save_dir $destfolder \
                    --log 1 --checkpoint $sourcefolder/checkpoints/ssl_training/wsi_encoder.pt \
                    --direct 0
                    --num_samples 10
                    --num_epochs 10