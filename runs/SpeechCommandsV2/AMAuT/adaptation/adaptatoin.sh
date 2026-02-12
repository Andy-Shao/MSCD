#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# Teachers knowledge distillation
# python -m runs.SpeechCommandsV2.AMAuT.adaptation.teacher_adaptation --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_level 'L2' --max_epoch 20 --lr '1e-4' --num_of_shft 5 --lr_gamma 30 \
#     --aut_lr_decay 0.55 --forbid_ls 'TST' --fail_coll_lim 6 \
#     --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.5}' \
#     --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA' --wandb

# Teachers teach student
# python -m runs.SpeechCommandsV2.AMAuT.adaptation.std_adapt --dataset 'SpeechCommandsV2' \
#     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
#     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
#     --corruption_level 'L2' --max_epoch 20 --lr '1e-4' \
#     --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.3, "END2":1.3, "ENSC":1.0, "PSH":1.0, "TST":1.5}' \
#     --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
#     --adpt_wght_pth './result/SpeechCommandsV2/AMAuT/Teach-Adapt' --wandb

# Teacher-student adaptation
python -m runs.SpeechCommandsV2.AMAuT.adaptation.teach_std_adapt --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.3, "END2":1.3, "ENSC":1.0, "PSH":1.0, "TST":1.5}' \
    --corruption_level 'L2' --max_epoch 20 --lr '1e-5' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/Std-Adapt' \
    --adpt_wght_pth './result/SpeechCommandsV2/AMAuT/Teach-Adapt'