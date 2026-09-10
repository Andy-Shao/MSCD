#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=98765
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='PAN_VS-C_L1-05.csv'
export ANAL_FILE_L2='PAN_VS-C_L2-05.csv'
export DATASET='VS-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.PANNs.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_level 'L2' --batch_size 32 --max_epoch 10 --num_of_shft 3 --fail_coll_lim 3 \
    --elect_weights '{"WHN":1.0, "ENQ":1.5, "END1":3.0, "END2":3.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --lrs '{"WHN":1e-4, "ENQ":1e-4, "END1":1e-4, "END2":1e-4, "ENSC":1e-4, "PSH":2e-4, "TST":1e-4}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/PANNs/TTDA' --forbid_ls 'END1' --freeze_pan --max_mode --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.PANNs.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_level 'L1' --batch_size 32 --max_epoch 7 --num_of_shft 3 --fail_coll_lim 2 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":3.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --lrs '{"WHN":1e-4, "ENQ":1e-4, "END1":1e-4, "END2":1e-4, "ENSC":1e-4, "PSH":2e-4, "TST":1e-4}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/PANNs/TTDA' --freeze_pan --forbid_ls 'END1' --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m ablation_study.No_CE.VocalSound.PANNs.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_level 'L2' --batch_size 32 --max_epoch 40 --lr 5e-4 --pseudo_threshold 8.85 \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 --lr_cardinality 60 \
    --elect_weights '{"WHN":1.0, "ENQ":1.5, "END1":3.0, "END2":3.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/PANNs/train' \
    --adpt_wght_pth './result/VocalSound/PANNs/Teach-Cons' \
    --max_mode --freeze_pan --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m ablation_study.No_CE.VocalSound.PANNs.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --corruption_level 'L1' --batch_size 32 --max_epoch 40 --lr 5e-4 --pseudo_threshold 6.92 \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 --lr_cardinality 60 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":3.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/PANNs/train' \
    --adpt_wght_pth './result/VocalSound/PANNs/Teach-Cons' \
    --max_mode --freeze_pan --seed $SEED_VAL

printf 'Student performance analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.PANNs.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/PANNs/train' \
    --std_adpt_wght_pth './result/VocalSound/PANNs/KD' \
    --output_path './result/VocalSound/PANNs/ablation_study/No_CE'

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.PANNs.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/PANNs/train' \
    --std_adpt_wght_pth './result/VocalSound/PANNs/KD' \
    --output_path './result/VocalSound/PANNs/ablation_study/No_CE'

printf 'ALL processing is finished\n' >> $LOG_FILE