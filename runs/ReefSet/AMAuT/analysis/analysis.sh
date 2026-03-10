#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.ReefSet.AMAuT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --output_file 'AuT_RS-C_L2.csv' \
    --std_adpt_wght_pth './result/ReefSet/AMAuT/Std-Adapt'