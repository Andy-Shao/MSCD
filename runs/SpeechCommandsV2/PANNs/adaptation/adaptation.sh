#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Adaptation
python -m runs.SpeechCommandsV2.PANNs.adaptation.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --corruption_type 'WHN' --corruption_level 'L2' --batch_size 64 --max_epoch 20 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --mse_rate 0.1 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.1 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb