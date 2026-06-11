#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Student Analysis
python -m runs.SpeechCommandsV2.AMAuT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --batch_size 32 --output_file 'AuT_SC2-C_L2.csv' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD'

python -m runs.SpeechCommandsV2.AMAuT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L1' --batch_size 32 --output_file 'AuT_SC2-C_L1.csv' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD'

# Silhouette Analysis
# python -m runs.SpeechCommandsV2.AMAuT.analysis.silhouette_anal --dataset 'SpeechCommandsV2' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_level 'L2' --batch_size 32 --output_file 'silhouette_AuT_SC2-C_L2.csv' \
#     --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
#     --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD'

# python -m runs.SpeechCommandsV2.AMAuT.analysis.silhouette_anal --dataset 'SpeechCommandsV2' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_level 'L1' --batch_size 32 --output_file 'silhouette_AuT_SC2-C_L1.csv' \
#     --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
#     --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD'