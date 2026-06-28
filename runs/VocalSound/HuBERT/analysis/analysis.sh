#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Student Analysis
python -m runs.VocalSound.HuBERT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file 'HuB_VS-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --std_adpt_wght_pth './result/VocalSound/HuBERT/KD'

python -m runs.VocalSound.HuBERT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file 'HuB_VS-C_L1.csv' --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --std_adpt_wght_pth './result/VocalSound/HuBERT/KD'

# Silhouette Analysis
# python -m runs.VocalSound.HuBERT.analysis.silhouette_anal --dataset 'VocalSound' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --output_file 'silhouette_HuB_VS-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
#     --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
#     --std_adpt_wght_pth './result/VocalSound/HuBERT/KD'