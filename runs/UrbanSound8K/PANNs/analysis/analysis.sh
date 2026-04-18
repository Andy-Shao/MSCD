#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Student performance analysis
# python -m runs.UrbanSound8K.PANNs.analysis.std_anal --dataset 'UrbanSound8K' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --output_file 'PAN_US8-C_L2.csv' --batch_size 32 --corruption_level 'L2' \
#     --orig_wght_pth './result/UrbanSound8K/PANNs/train' \
#     --std_adpt_wght_pth './result/UrbanSound8K/PANNs/KD'

# Silhouette analysis
python -m runs.UrbanSound8K.PANNs.analysis.silhouette_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file 'silhouette_PAN_US8-C_L2.csv' --batch_size 32 --corruption_level 'L2' \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/PANNs/KD'