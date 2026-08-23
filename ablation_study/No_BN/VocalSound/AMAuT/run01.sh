#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=2026
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='AuT_VS-C_L1.csv'
export ANAL_FILE_L2='AuT_VS-C_L2.csv'
export DATASET='VS-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.AMAuT.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 3 --fail_coll_lim 3 \
    --max_epoch 30 --lr '1e-5' --lr_cardinality 80 --aut_lr_decay 0.55 \
    --elect_weights '{"WHN":1.3, "ENQ":1.0, "END1":1.3, "END2":1.3, "ENSC":1.0, "PSH":0.9, "TST":1.3}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.AMAuT.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L1' --num_of_shft 3 --fail_coll_lim 3 \
    --max_epoch 30 --lr '1e-5' --lr_cardinality 80 \
    --elect_weights '{"WHN":1.3, "ENQ":1.0, "END1":1.3, "END2":1.0, "ENSC":1.3, "PSH":1.0, "TST":1.3}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/TTDA' --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.AMAuT.adaptation.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 20 --lr '1e-4' --ctr_rt 1.0 \
    --pseudo_threshold 6.23 --ctr_dist 'sq_l2' --lr_cardinality 40 \
    --elect_weights '{"WHN":1.3, "ENQ":1.0, "END1":1.3, "END2":1.3, "ENSC":1.0, "PSH":0.9, "TST":1.3}' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/train' \
    --adpt_wght_pth './result/VocalSound/AMAuT/Teach-Cons' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.AMAuT.adaptation.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 20 --lr '1e-4' --ctr_rt 1.0 \
    --pseudo_threshold 6.31 --ctr_dist 'sq_l2' --lr_cardinality 40 \
    --elect_weights '{"WHN":1.3, "ENQ":1.0, "END1":1.3, "END2":1.0, "ENSC":1.3, "PSH":1.0, "TST":1.3}' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/train' \
    --adpt_wght_pth './result/VocalSound/AMAuT/Teach-Cons' --seed $SEED_VAL

printf 'Student Model Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.AMAuT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L2 --corruption_level 'L2' --batch_size 32 \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/train' \
    --std_adpt_wght_pth './result/VocalSound/AMAuT/KD' \
    --output_path './result/VocalSound/AMAuT/ablation_study/No_BN'

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.AMAuT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L1 --corruption_level 'L1' --batch_size 32 \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/train' \
    --std_adpt_wght_pth './result/VocalSound/AMAuT/KD' \
    --output_path './result/VocalSound/AMAuT/ablation_study/No_BN'

printf 'ALL processing is finished\n' >> $LOG_FILE