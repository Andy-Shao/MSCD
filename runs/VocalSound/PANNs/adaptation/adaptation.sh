#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Adaptation
python -m runs.VocalSound.PANNs.adaptation.ttda --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_type 'WHN' --corruption_level 'L2' --batch_size 32 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --gent_q 1.6 --mse_rate 0.0 \
    --orig_wght_pth './result/VocalSound/PANNs/train' --freeze_pan --wandb