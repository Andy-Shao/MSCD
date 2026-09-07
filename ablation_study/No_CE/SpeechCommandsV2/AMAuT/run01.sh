#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=2026
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='AuT_SC2-C_L1.csv'
export ANAL_FILE_L2='AuT_SC2-C_L2.csv'
export DATASET='SC2-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --max_epoch 30 --lr '1e-4' --num_of_shft 5 --lr_gamma 10 \
    --aut_lr_decay 0.55 --forbid_ls 'TST' --fail_coll_lim 6 --unfrz_pos 22 \
    --lr_threshold 10 --lr_cardinality 60 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":2.0}' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L1' --max_epoch 30 --lr 1e-4 --num_of_shft 3 --fail_coll_lim 3 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.5, "END2":1.5, "ENSC":1.0, "PSH":1.0, "TST":3.0}' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/TTDA' --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m ablation_study.No_CE.SpeechCommandsV2.AMAuT.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --max_epoch 30 --lr '1e-4' --pseudo_threshold 6.15 \
    --lr_threshold 10 --ctr_rt 1.0 --ctr_dist 'sq_l2' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":2.0}' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_pth './result/SpeechCommandsV2/AMAuT/Teach-Cons' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m ablation_study.No_CE.SpeechCommandsV2.AMAuT.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' --batch_size 32 \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L1' --max_epoch 20 --lr '1e-4' --pseudo_threshold 7.69 \
    --lr_threshold 10 --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.5, "END2":1.5, "ENSC":1.0, "PSH":1.0, "TST":3.0}' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --adpt_wght_pth './result/SpeechCommandsV2/AMAuT/Teach-Cons' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L2' --batch_size 32 --output_file $ANAL_FILE_L2 \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD' \
    --output_path './result/SpeechCommandsV2/AMAuT/ablation_study/No_CE'

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.AMAuT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --corruption_level 'L1' --batch_size 32 --output_file $ANAL_FILE_L1 \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/AMAuT/KD' \
    --output_path './result/SpeechCommandsV2/AMAuT/ablation_study/No_CE'

printf 'ALL processing is finished\n' >> $LOG_FILE