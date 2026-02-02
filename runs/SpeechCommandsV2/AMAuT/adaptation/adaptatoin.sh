#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommandsV2.AMAuT.adaptation.teacher_adaptation --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --corruption_level 'L2' \
    --elect_weights '{"WHN":0.8855, "ENQ":0.8790, "END1":0.9222, "END2":0.9162, "ENSC":0.8489, "PSH":0.7040, "TST":0.9443}' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_path $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA'