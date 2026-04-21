#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.VocalSound.PANNs.train.train --dataset 'VocalSound' \
    --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --batch_size 32 --lr 1e-4 --max_epoch 20 --wandb