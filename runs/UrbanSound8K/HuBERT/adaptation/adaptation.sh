#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# python -m runs.UrbanSound8K.HuBERT.adaptation.teach_cons --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --batch_size 32 --corruption_level 'L2' --num_of_shft 1 --fail_coll_lim 2 --max_epoch 30 --lr '1e-5' \
#     --lr_cardinality 80 --lr_threshold 35 \
#     --elect_weights '{"WHN":1.0, "ENSC":0.83, "PSH":1.0, "TST":1.0}' \
#     --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.9, "TST":0.9}' \
#     --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/TTDA' --wandb

python -m runs.UrbanSound8K.HuBERT.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L1' --num_of_shft 1 --fail_coll_lim 2 --max_epoch 9 --lr '1e-5' \
    --lr_cardinality 80 --lr_threshold 35 \
    --elect_weights '{"WHN":0.9, "ENSC":0.9, "PSH":1.2, "TST":1.1}' \
    --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.70, "TST":0.9}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/TTDA' --wandb

# python -m runs.UrbanSound8K.HuBERT.adaptation.kd --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --batch_size 32 --corruption_level 'L2' --max_epoch 35 --lr '1e-4' \
#     --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 2.95 \
#     --elect_weights '{"WHN":1.0, "ENSC":0.83, "PSH":1.0, "TST":1.0}' \
#     --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
#     --adpt_wght_pth './result/UrbanSound8K/HuBERT/Teach-Cons' --wandb