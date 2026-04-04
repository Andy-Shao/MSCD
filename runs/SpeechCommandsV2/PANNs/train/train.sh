#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommandsV2.PANNs.train.train --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/data' \
    