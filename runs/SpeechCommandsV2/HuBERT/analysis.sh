#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommandsV2.HuBERT.noise_supression_analysis --dataset 'SpeechCommandsV2' --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --use_pre_trained_weigth --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train'