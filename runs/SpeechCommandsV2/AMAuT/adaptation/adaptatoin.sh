#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

# {"WHN":0.8855, "ENQ":0.8790, "END1":0.9222, "END2":0.9162, "ENSC":0.8489, "PSH":0.7040, "TST":0.9443}
python -m runs.SpeechCommandsV2.AMAuT.adaptation.teacher_adaptation --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --max_epoch 10 --lr '1e-4' --num_of_shft 5 --lr_gamma 30 \
    --aut_lr_decay 0.55 --forbid_ls 'TST'\
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.5}' \
    --adpt_wght_path $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA' --wandb