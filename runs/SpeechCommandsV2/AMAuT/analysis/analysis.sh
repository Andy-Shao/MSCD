#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommandsV2.AMAuT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --batch_size 32 --output_file 'AuT_SC2-C_L2.csv' \
    --elect_weights '{"WHN":2.0, "ENQ":2.0, "END1":2.0, "END2":2.0, "ENSC":1.5, "PSH":1.5, "TST":2.0}' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/Std-Adapt'