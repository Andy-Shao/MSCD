#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Student Analysis
python -m runs.SpeechCommandsV2.HuBERT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file 'HuB_SC2-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD'

python -m runs.SpeechCommandsV2.HuBERT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file 'HuB_SC2-C_L1.csv' --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD'

# Silhouette Analysis
# python -m runs.SpeechCommandsV2.HuBERT.analysis.silhouette_anal --dataset 'SpeechCommandsV2' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --output_file 'silhouette_HuB_SC2-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
#     --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' \
#     --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD'

# python -m runs.SpeechCommandsV2.HuBERT.analysis.silhouette_anal --dataset 'SpeechCommandsV2' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --output_file 'silhouette_HuB_SC2-C_L1.csv' --batch_size 32 --corruption_level 'L1' --model_level 'base' \
#     --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' \
#     --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD'