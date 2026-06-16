#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Student Analysis
python -m runs.UrbanSound8K.HuBERT.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file 'HuB_US8-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/HuBERT/KD'

python -m runs.UrbanSound8K.HuBERT.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file 'HuB_US8-C_L1.csv' --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/HuBERT/KD'

# Silhouette Analysis
# python -m runs.UrbanSound8K.HuBERT.analysis.silhouette_anal --dataset 'UrbanSound8K' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --output_file 'silhouette_HuB_US8-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
#     --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
#     --std_adpt_wght_pth './result/UrbanSound8K/HuBERT/KD'