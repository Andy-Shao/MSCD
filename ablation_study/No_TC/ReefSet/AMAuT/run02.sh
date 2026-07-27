#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=12345
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='AuT_RS-C_L1-02.csv'
export ANAL_FILE_L2='AuT_RS-C_L2-02.csv'
export DATASET='RS-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.ReefSet.AMAuT.adaptation.kd --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 10 --lr 1e-4 --pseudo_threshold 7.85 \
    --ctr_rt 0.5 --ctr_dist 'sq_l2' \
    --elect_weights '{"WHN":1.1, "ENQ":1.0, "END1":2.0, "END2":1.1, "ENSC":1.0, "PSH":1.0, "TST":3.0}' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/TTDA' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.ReefSet.AMAuT.adaptation.kd --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 20 --lr 1e-4 --pseudo_threshold 8.85 \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 2.0 --lr_momentum 0.75 \
    --elect_weights '{"WHN":1.5, "ENQ":1.0, "END1":2.0, "END2":1.0, "ENSC":1.0, "PSH":2.0, "TST":3.0}' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/TTDA' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.ReefSet.AMAuT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --output_file $ANAL_FILE_L2 \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' \
    --std_adpt_wght_pth './result/ReefSet/AMAuT/KD' \
    --output_path './result/ReefSet/AMAuT/ablation_study/No_TC'

printf 'L1\n' >> $LOG_FILE
python -m runs.ReefSet.AMAuT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L1' --output_file $ANAL_FILE_L1 \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/train' \
    --std_adpt_wght_pth './result/ReefSet/AMAuT/KD' \
    --output_path './result/ReefSet/AMAuT/ablation_study/No_TC'

printf 'ALL processing is finished\n' >> $LOG_FILE