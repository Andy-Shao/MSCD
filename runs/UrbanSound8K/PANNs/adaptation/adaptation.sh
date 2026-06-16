#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
# python -m runs.UrbanSound8K.PANNs.adaptation.teach_cons --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --corruption_level 'L2' --batch_size 33 --max_epoch 10 --lr 1e-5 \
#     --num_of_shft 1 --fail_coll_lim 2 --lr_gamma 30 \
#     --elect_weights '{"WHN":1.5, "ENSC":0.8, "PSH":1.07, "TST":1.5}' \
#     --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.9, "TST":0.9}' \
#     --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/TTDA' --max_mode --wandb

python -m runs.UrbanSound8K.PANNs.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --corruption_level 'L1' --batch_size 33 --max_epoch 10 --lr 1e-4 \
    --num_of_shft 1 --fail_coll_lim 2 --lr_gamma 30 \
    --elect_weights '{"WHN":1.5, "ENSC":1.0, "PSH":1.0, "TST":1.5}' \
    --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.9, "TST":0.9}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/TTDA' --max_mode --wandb

# Knowledge Distillation
# python -m runs.UrbanSound8K.PANNs.adaptation.kd --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --corruption_level 'L2' --batch_size 33 --max_epoch 30 --lr 1e-4 \
#     --pseudo_threshold 3.75 --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 \
#     --elect_weights '{"WHN":1.5, "ENSC":0.8, "PSH":1.07, "TST":1.5}' \
#     --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/train' \
#     --adpt_wght_pth './result/UrbanSound8K/PANNs/Teach-Cons' --wandb