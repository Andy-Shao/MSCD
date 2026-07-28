#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=12345
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='HuB_RS-C_L1-02.csv'
export ANAL_FILE_L2='HuB_RS-C_L2-02.csv'
export DATASET='RS-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.ReefSet.HuBERT.adaptation.kd --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 10 --pseudo_threshold 7.77 --ctr_rt 1.0 \
    --ctr_dist 'sq_l2' --ctr_T 1.0 --lr 1e-4 --model_level 'base' \
    --elect_weights '{"WHN":1.0, "ENQ":0.8, "END1":1.0, "END2":1.0, "ENSC":0.8, "PSH":2.5, "TST":3.0}' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/train' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.ReefSet.HuBERT.adaptation.kd --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 15 --pseudo_threshold 9.15 --ctr_rt 1.0 \
    --ctr_dist 'sq_l2' --ctr_T 1.0 --lr 1e-4 --model_level 'base' \
    --elect_weights '{"WHN":0.9, "ENQ":1.0, "END1":1.5, "END2":1.0, "ENSC":1.0, "PSH":3.5, "TST":3.0}' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/train' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/TTDA' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.ReefSet.HuBERT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --std_adpt_wght_pth './result/ReefSet/HuBERT/KD' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/train' \
    --output_path './result/ReefSet/HuBERT/ablation_study/No_TC'

printf 'L1\n' >> $LOG_FILE
python -m runs.ReefSet.HuBERT.analysis.std_anal --dataset 'ReefSet' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --std_adpt_wght_pth './result/ReefSet/HuBERT/KD' \
    --orig_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/train' \
    --output_path './result/ReefSet/HuBERT/ablation_study/No_TC'

printf 'ALL processing is finished\n' >> $LOG_FILE