#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Adaptation
# python -m runs.UrbanSound8K.PANNs.adaptation.ttda --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --corruption_type 'WHN' --corruption_level 'L2' --batch_size 70 --max_epoch 25 --lr 1e-4 \
#     --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
#     --freeze_pan \
#     --orig_wght_pth './result/UrbanSound8K/PANNs/train' --wandb

python -m runs.UrbanSound8K.PANNs.adaptation.ttda --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --corruption_type 'ENSC' --corruption_level 'L2' --batch_size 70 --max_epoch 45 --lr 1e-4 \
    --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 1.6 --mse_rate 0.1 \
    --freeze_pan --lr_cardinality 100 --lr_threshold 20 \
    --orig_wght_pth './result/UrbanSound8K/PANNs/train' --wandb