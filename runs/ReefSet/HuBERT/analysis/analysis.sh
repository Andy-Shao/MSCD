#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Student Analysis
python -m runs.ReefSet.HuBERT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file 'HuB_RS-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --std_adpt_wght_pth './result/ReefSet/HuBERT/KD' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/train'

python -m runs.ReefSet.HuBERT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file 'HuB_RS-C_L1.csv' --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --std_adpt_wght_pth './result/ReefSet/HuBERT/KD' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/train'

# Silhouette Analysis
# python -m runs.ReefSet.HuBERT.analysis.silhouette_anal --dataset 'ReefSet' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --output_file 'silhouette_HuB_RS-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
#     --std_adpt_wght_pth './result/ReefSet/HuBERT/KD' \
#     --orig_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/train'