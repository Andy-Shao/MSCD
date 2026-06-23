#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
# python -m runs.ReefSet.PANNs.adaptation.teach_cons --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 32 --corruption_level 'L2' --num_of_shft 3 --fail_coll_lim 5 --max_epoch 20 \
#     --elect_weights '{"WHN":1.0, "ENQ":0.8, "END1":1.0, "END2":0.9, "ENSC":0.8, "PSH":3.0, "TST":1.0}' \
#     --lrs '{"WHN":1e-5, "ENQ":1e-4, "END1":1e-4, "END2":1e-5, "ENSC":1e-5, "PSH":1e-4, "TST":1e-4}' \
#     --pan_lr_decaies '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":0.55, "PSH":1.0, "TST":1.0}' \
#     --adpt_wght_pth $BASE_PATH'/result/ReefSet/PANNs/TTDA' --freeze_pan --wandb

# python -m runs.ReefSet.PANNs.adaptation.teach_cons --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 32 --corruption_level 'L1' --num_of_shft 3 --fail_coll_lim 3 --max_epoch 20 \
#     --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":3.0, "TST":1.5}' \
#     --lr_momentums '{"WHN":0.9, "ENQ":0.9, "END1":0.9, "END2":0.9, "ENSC":0.75, "PSH":0.9, "TST":0.9}' \
#     --adpt_wght_pth $BASE_PATH'/result/ReefSet/PANNs/TTDA' --wandb

# Knowledge Distillation
# python -m runs.ReefSet.PANNs.adaptation.kd --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 32 --corruption_level 'L2' --max_epoch 20 --pseudo_threshold 6.54 --ctr_rt 1.0 \
#     --ctr_dist 'sq_l2' --ctr_T 1.0 --lr 1e-4 --lr_momentum 0.75 \
#     --elect_weights '{"WHN":1.0, "ENQ":0.8, "END1":1.0, "END2":0.9, "ENSC":0.8, "PSH":3.0, "TST":1.0}' \
#     --orig_wght_pth $BASE_PATH'/result/ReefSet/PANNs/train' \
#     --adpt_wght_pth './result/ReefSet/PANNs/Teach-Cons' --freeze_pan --wandb

python -m runs.ReefSet.PANNs.adaptation.kd --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 20 --pseudo_threshold 7.31 --ctr_rt 1.0 \
    --ctr_dist 'sq_l2' --ctr_T 1.0 --lr 1e-4 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":3.0, "TST":1.5}' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/PANNs/train' \
    --adpt_wght_pth './result/ReefSet/PANNs/Teach-Cons' --freeze_pan --wandb