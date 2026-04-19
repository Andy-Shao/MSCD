#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Student Analysis
# python -m runs.ReefSet.AMAuT.analysis.std_anal --dataset 'ReefSet' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 32 --corruption_level 'L2' --output_file 'AuT_RS-C_L2.csv' \
#     --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' \
#     --std_adpt_wght_pth './result/ReefSet/AMAuT/KD'

# Silhouette Analysis
python -m runs.ReefSet.AMAuT.analysis.silhouette_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --output_file 'silhouette_AuT_RS-C_L2.csv' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' \
    --std_adpt_wght_pth './result/ReefSet/AMAuT/KD'