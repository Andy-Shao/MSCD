#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommands.noise_supression --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C'