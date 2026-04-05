#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommandsV2.PANNs.train.train --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/data' \
    --batch_size 32 --lr '5e-4' --max_epoch 20 --wandb