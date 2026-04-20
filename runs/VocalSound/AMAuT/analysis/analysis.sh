#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Student Model Analysis
python -m runs.VocalSound.AMAuT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file 'AuT_VS-C_L2.csv' --corruption_level 'L2' --batch_size 32 \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/train' \
    --std_adpt_wght_pth './result/VocalSound/AMAuT/KD'