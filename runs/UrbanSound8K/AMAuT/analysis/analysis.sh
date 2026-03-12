#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.UrbanSound8K.AMAuT.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file 'AuT_US8-C_L2.csv' --corruption_level 'L2' --batch_size 32 \
    --std_adpt_wght_pth './result/UrbanSound8K/AMAuT/KD'