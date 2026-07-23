#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=98765
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='AuT_SC2-C_L1-05.csv'
export ANAL_FILE_L2='AuT_SC2-C_L2-05.csv'
export DATASET='SC2-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.adaptation.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --max_epoch 30 --lr '1e-4' --pseudo_threshold 6.15 \
    --lr_threshold 10 --ctr_rt 1.0 --ctr_dist 'sq_l2' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":2.0}' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.adaptation.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L1' --max_epoch 20 --lr '1e-4' --pseudo_threshold 7.69 \
    --lr_threshold 10 --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.5, "END2":1.5, "ENSC":1.0, "PSH":1.0, "TST":3.0}' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --batch_size 32 --output_file $ANAL_FILE_L2 \
    --output_path './result/SpeechCommandsV2/AMAuT/ablation_study/No_TC' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD'

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L1' --batch_size 32 --output_file $ANAL_FILE_L1 \
    --output_path './result/SpeechCommandsV2/AMAuT/ablation_study/No_TC' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD'

printf 'ALL processing is finished\n' >> $LOG_FILE