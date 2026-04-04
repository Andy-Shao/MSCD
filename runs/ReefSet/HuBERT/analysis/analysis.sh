#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.ReefSet.HuBERT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file 'HuB_RS-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --std_adpt_wght_pth './result/ReefSet/HuBERT/KD' \
    --orig_wght_pth './result/ReefSet/HuBERT/train'