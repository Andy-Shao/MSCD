#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=98765
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='HuB_US8-C_L1-05.csv'
export ANAL_FILE_L2='HuB_US8-C_L2-05.csv'
export DATASET='US8-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.UrbanSound8K.HuBERT.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 1 --fail_coll_lim 2 --max_epoch 30 --lr '1e-5' \
    --lr_cardinality 80 --lr_threshold 35 \
    --elect_weights '{"WHN":1.0, "ENSC":0.83, "PSH":1.0, "TST":1.0}' \
    --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.9, "TST":0.9}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.UrbanSound8K.HuBERT.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L1' --num_of_shft 1 --fail_coll_lim 2 --max_epoch 9 --lr '1e-5' \
    --lr_cardinality 80 --lr_threshold 35 \
    --elect_weights '{"WHN":0.9, "ENSC":0.9, "PSH":1.2, "TST":1.1}' \
    --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.70, "TST":0.9}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/TTDA' --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L1\n' >> $LOG_FILE
python -m runs.UrbanSound8K.HuBERT.adaptation.kd --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 35 --lr '1e-4' \
    --ctr_rt 0.0 --ctr_dist 'sq_l2' --pseudo_threshold 2.95 \
    --elect_weights '{"WHN":1.0, "ENSC":0.83, "PSH":1.0, "TST":1.0}' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
    --adpt_wght_pth './result/UrbanSound8K/HuBERT/Teach-Cons' --seed $SEED_VAL

printf 'L2\n' >> $LOG_FILE
python -m runs.UrbanSound8K.HuBERT.adaptation.kd --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 35 --lr '1e-4' \
    --ctr_rt 0.0 --ctr_dist 'sq_l2' --pseudo_threshold 3.15 \
    --elect_weights '{"WHN":0.9, "ENSC":0.9, "PSH":1.2, "TST":1.1}' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
    --adpt_wght_pth './result/UrbanSound8K/HuBERT/Teach-Cons' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.UrbanSound8K.HuBERT.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/HuBERT/KD' \
    --output_path './result/UrbanSound8K/HuBERT/ablation_study/No_CD'

printf 'L1\n' >> $LOG_FILE
python -m runs.UrbanSound8K.HuBERT.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/HuBERT/KD' \
    --output_path './result/UrbanSound8K/HuBERT/ablation_study/No_CD'

printf 'ALL processing is finished\n' >> $LOG_FILE