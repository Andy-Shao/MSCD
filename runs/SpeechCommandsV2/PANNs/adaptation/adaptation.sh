#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Adaptation
# python -m runs.SpeechCommandsV2.PANNs.adaptation.ttda --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_type 'WHN' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --mse_rate 0.1 \
#     --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.1 \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb

# python -m runs.SpeechCommandsV2.PANNs.adaptation.ttda --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_type 'ENQ' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
#     --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb

# python -m runs.SpeechCommandsV2.PANNs.adaptation.ttda --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_type 'END1' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
#     --lr 5e-5 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
#     --lr_momentum 0.70 --pan_lr_decay 1.0 --gent_q 1.6 \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb

# python -m runs.SpeechCommandsV2.PANNs.adaptation.ttda --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_type 'END2' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
#     --lr_momentum 0.75 --pan_lr_decay 1.0 --gent_q 1.6 \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb

# python -m runs.SpeechCommandsV2.PANNs.adaptation.ttda --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_type 'ENSC' --corruption_level 'L2' --batch_size 70 --max_epoch 25 \
#     --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --mse_rate 0.1 \
#     --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.6 \
#     --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb

python -m runs.SpeechCommandsV2.PANNs.adaptation.ttda --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_type 'PSH' --corruption_level 'L2' --batch_size 70 --max_epoch 60 \
    --lr 1e-4 --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.0 --mse_rate 0.0 \
    --lr_momentum 0.9 --pan_lr_decay 1.0 --gent_q 1.1 --lr_cardinality 80 \
    --orig_wght_pth './result/SpeechCommandsV2/PANNs/train' --wandb