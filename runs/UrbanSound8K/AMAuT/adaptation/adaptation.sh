#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
# python -m runs.UrbanSound8K.AMAuT.adaptation.teach_cons --dataset 'UrbanSound8K' \
#     --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
#     --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
#     --batch_size 32 --corruption_level 'L2' --num_of_shft 2 --fail_coll_lim 2 --max_epoch 30 \
#     --lr '1e-4' --lr_cardinality 60 \
#     --elect_weights '{"WHN":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
#     --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/TTDA' --wandb

# Knowledge Distillation
python -m runs.UrbanSound8K.AMAuT.adaptation.kd --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 20 --lr '1e-4' \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 3.1 \
    --elect_weights '{"WHN":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/train' \
    --adpt_wght_pth './result/UrbanSound8K/AMAuT/Teach-Cons' --wandb