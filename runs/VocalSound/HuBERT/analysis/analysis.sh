#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.VocalSound.HuBERT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file 'HuB_VS-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --std_adpt_wght_pth './result/VocalSound/HuBERT/KD'