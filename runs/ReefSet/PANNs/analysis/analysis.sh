#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Student performance analysis
python -m runs.ReefSet.PANNs.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file 'PAN_RS-C_L2.csv' --batch_size 32 --corruption_level 'L2' \
    --orig_wght_pth './result/ReefSet/PANNs/train' \
    --std_adpt_wght_pth './result/ReefSet/PANNs/KD'