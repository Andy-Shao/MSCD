#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
python -m runs.ReefSet.AMAuT.adaptation.teach_cons --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 20 \
    --forbid_ls 'TST' --lr_gamma 30 --num_of_shft 3 --fail_coll_lim 3 \
    --elect_weights '{"WHN":1.1, "ENQ":1.0, "END1":2.0, "END2":1.1, "ENSC":1.0, "PSH":1.0, "TST":3.0}' \
    --lrs '{"WHN":1e-4, "ENQ":1e-4, "END1":1e-5, "END2":1e-5, "ENSC":1e-4, "PSH":1e-4, "TST":1e-4}' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/TTDA' --wandb

# python -m runs.ReefSet.AMAuT.adaptation.teach_cons --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 32 --corruption_level 'L1' --max_epoch 15 --num_of_shft 3 --fail_coll_lim 3 \
#     --elect_weights '{"WHN":1.5, "ENQ":1.0, "END1":2.0, "END2":1.0, "ENSC":1.0, "PSH":2.0, "TST":3.0}' \
#     --lrs '{"WHN":1e-5, "ENQ":1e-4, "END1":1e-5, "END2":1e-4, "ENSC":1e-4, "PSH":1e-5, "TST":1e-4}' \
#     --adpt_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/TTDA' --wandb

# Knowledge Distillation
# python -m runs.ReefSet.AMAuT.adaptation.kd --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 32 --corruption_level 'L2' --max_epoch 20 --lr 1e-4 --pseudo_threshold 7.85 \
#     --ctr_rt 0.0 --ctr_dist 'sq_l2' \
#     --elect_weights '{"WHN":1.1, "ENQ":1.0, "END1":2.0, "END2":1.1, "ENSC":1.0, "PSH":1.0, "TST":3.0}' \
#     --adpt_wght_pth './result/ReefSet/AMAuT/Teach-Cons' \
#     --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' --wandb

# python -m runs.ReefSet.AMAuT.adaptation.kd --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 32 --corruption_level 'L1' --max_epoch 20 --lr 1e-4 --pseudo_threshold 8.85 \
#     --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 \
#     --elect_weights '{"WHN":1.5, "ENQ":1.0, "END1":2.0, "END2":1.0, "ENSC":1.0, "PSH":2.0, "TST":3.0}' \
#     --adpt_wght_pth './result/ReefSet/AMAuT/Teach-Cons' \
#     --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' --wandb