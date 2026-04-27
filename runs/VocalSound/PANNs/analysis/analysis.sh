#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Student performance analysis
python -m runs.VocalSound.PANNs.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file 'PAN_VS-C_L2.csv' --batch_size 32 --corruption_level 'L2' \
    --orig_wght_pth './result/VocalSound/PANNs/train' \
    --std_adpt_wght_pth './result/VocalSound/PANNs/KD'