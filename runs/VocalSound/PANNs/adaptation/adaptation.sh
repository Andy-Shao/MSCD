#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
# python -m runs.VocalSound.PANNs.adaptation.teach_cons --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_level 'L2' --batch_size 32 --max_epoch 14 --num_of_shft 3 --fail_coll_lim 3 \
#     --elect_weights '{"WHN":1.0, "ENQ":1.5, "END1":3.0, "END2":3.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
#     --lrs '{"WHN":1e-4, "ENQ":1e-4, "END1":1e-4, "END2":1e-4, "ENSC":1e-4, "PSH":2e-4, "TST":1e-4}' \
#     --adpt_wght_pth $BASE_PATH'/result/VocalSound/PANNs/TTDA' --forbid_ls 'END1' --freeze_pan --wandb

python -m runs.VocalSound.PANNs.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_level 'L1' --batch_size 32 --max_epoch 7 --num_of_shft 3 --fail_coll_lim 2 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":3.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --lrs '{"WHN":1e-4, "ENQ":1e-4, "END1":1e-4, "END2":1e-4, "ENSC":1e-4, "PSH":2e-4, "TST":1e-4}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/PANNs/TTDA' --freeze_pan --forbid_ls 'END1' --wandb

# Knowledge Distillation
# python -m runs.VocalSound.PANNs.adaptation.kd --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --corruption_level 'L2' --batch_size 32 --max_epoch 40 --lr 5e-4 --pseudo_threshold 7.38 \
#     --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 --lr_cardinality 60 \
#     --elect_weights '{"WHN":1.5, "ENQ":1.5, "END1":2.0, "END2":2.0, "ENSC":1.0, "PSH":0.8, "TST":0.8}' \
#     --orig_wght_pth './result/VocalSound/PANNs/train' \
#     --adpt_wght_pth './result/VocalSound/PANNs/Teach-Cons' \
#     --max_mode --wandb