#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
# python -m runs.SpeechCommandsV2.AMAuT.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_level 'L2' --max_epoch 30 --lr '1e-4' --num_of_shft 5 --lr_gamma 10 \
#     --aut_lr_decay 0.55 --forbid_ls 'TST' --fail_coll_lim 6 --rewgt_pos 20 --unfrz_pos 22 \
#     --lr_threshold 10 --str_pos 0 --lr_cardinality 60 \
#     --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":2.0}' \
#     --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA' --wandb

# Knowledge distillation
python -m runs.SpeechCommandsV2.AMAuT.adaptation.std_adapt --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --max_epoch 30 --lr '1e-4' --pseudo_threshold 10.0 \
    --lr_threshold 10 --ctr_rt 1.0 --ctr_dist 'sq_l2' \
    --elect_weights '{"WHN":2.0, "ENQ":2.0, "END1":2.0, "END2":2.0, "ENSC":1.5, "PSH":1.5, "TST":2.0}' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_pth './result/SpeechCommandsV2/AMAuT/Teach-Adapt' --wandb