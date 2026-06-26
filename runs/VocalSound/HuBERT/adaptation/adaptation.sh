#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
python -m runs.VocalSound.HuBERT.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 4 --fail_coll_lim 3 --model_level 'base' \
    --max_epoch 20  --forbid_ls 'WHN' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.2, "END2":1.2, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --lrs '{"WHN":1e-5, "ENQ":1e-4, "END1":1e-4, "END2":1e-4, "ENSC":1e-4, "PSH":1e-5, "TST":1e-5}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/TTDA' --wandb

# Knowledge Distillation
# python -m runs.VocalSound.HuBERT.adaptation.kd --dataset 'VocalSound' \
#     --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
#     --eval_set_path $BASE_PATH'/data/VocalSound-C' \
#     --batch_size 32 --corruption_level 'L2' --model_level 'base' --max_epoch 25 --lr 1e-4 \
#     --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.2, "END2":1.2, "ENSC":1.0, "PSH":1.3, "TST":1.0}' \
#     --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 5.92 \
#     --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
#     --adpt_wght_pth './result/VocalSound/HuBERT/Teach-Cons' --wandb