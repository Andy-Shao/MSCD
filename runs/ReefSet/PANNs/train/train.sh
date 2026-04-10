#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.ReefSet.PANNs.train.train --dataset 'ReefSet' \
    --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --batch_size 33 --lr 1e-3 --max_epoch 20 --wandb