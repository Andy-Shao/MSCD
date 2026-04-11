#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Student performance analysis
python -m runs.SpeechCommandsV2.PANNs.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file 'PAN_SC2-C_L2.csv' --batch_size 32 --corruption_level 'L2' \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/PANNs/KD'

# Silhouette analysis
# python -m runs.SpeechCommandsV2.PANNs.analysis.silhouette_anal --dataset 'SpeechCommandsV2' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --output_file 'silhouette_PAN_SC2-C_L2.csv' --batch_size 32 --corruption_level 'L2' \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' \
#     --std_adpt_wght_pth './result/SpeechCommandsV2/PANNs/KD'