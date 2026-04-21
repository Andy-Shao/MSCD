#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.VocalSound.PANNs.train.train --dataset 'VocalSound' \
    --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --batch_size 32 --lr 5e-4 --max_epoch 40 --lr_cardinality 80 \
    --lr_threshold 20 --wandb