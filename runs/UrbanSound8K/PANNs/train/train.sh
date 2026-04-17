#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.UrbanSound8K.PANNs.train.train --dataset 'UrbanSound8K' \
    --dataset_root_path $BASE_PATH'/data/UrbanSound8K' \
    --batch_size 33 --lr 5e-4 --max_epoch 30 --wandb 