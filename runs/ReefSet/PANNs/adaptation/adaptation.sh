#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Adaptation
python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 70 --corruption_level 'L2' --corruption_type 'WHN' --lr 1e-4 --max_epoch 20 \
    --nucnm_rate 1.2 --ent_rate 0.0 --gent_rate 0.0 --gent_q 0.9 --mse_rate 0.1 \
    --orig_wght_pth './result/ReefSet/PANNs/train' --wandb