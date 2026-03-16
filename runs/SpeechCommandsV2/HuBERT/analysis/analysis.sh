#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommandsV2.HuBERT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file 'HuB_SC2-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD'