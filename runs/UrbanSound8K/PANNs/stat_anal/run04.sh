#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=56789
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='PAN_US8-C_L1-04.csv'
export ANAL_FILE_L2='PAN_US8-C_L2-04.csv'
export DATASET='US8-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.UrbanSound8K.PANNs.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --corruption_level 'L2' --batch_size 33 --max_epoch 10 --lr 1e-5 \
    --num_of_shft 1 --fail_coll_lim 2 --lr_gamma 30 \
    --elect_weights '{"WHN":1.5, "ENSC":0.8, "PSH":1.07, "TST":1.5}' \
    --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.9, "TST":0.9}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/TTDA' --max_mode --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.UrbanSound8K.PANNs.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --corruption_level 'L1' --batch_size 33 --max_epoch 10 --lr 8e-5 \
    --num_of_shft 1 --fail_coll_lim 2 --lr_gamma 30 \
    --elect_weights '{"WHN":1.5, "ENSC":1.0, "PSH":1.0, "TST":1.5}' \
    --lr_momentums '{"WHN":0.9, "ENSC":0.9, "PSH":0.9, "TST":0.9}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/TTDA' --max_mode --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.UrbanSound8K.PANNs.adaptation.kd --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --corruption_level 'L2' --batch_size 33 --max_epoch 30 --lr 1e-4 \
    --pseudo_threshold 3.75 --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 \
    --elect_weights '{"WHN":1.5, "ENSC":0.8, "PSH":1.07, "TST":1.5}' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/train' \
    --adpt_wght_pth './result/UrbanSound8K/PANNs/Teach-Cons' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.UrbanSound8K.PANNs.adaptation.kd --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --corruption_level 'L1' --batch_size 33 --max_epoch 30 --lr 1e-4 \
    --pseudo_threshold 3.85 --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 \
    --elect_weights '{"WHN":1.5, "ENSC":1.0, "PSH":1.0, "TST":1.5}' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/train' \
    --adpt_wght_pth './result/UrbanSound8K/PANNs/Teach-Cons' --seed $SEED_VAL

printf 'Student performance analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.UrbanSound8K.PANNs.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/PANNs/KD'

printf 'L1\n' >> $LOG_FILE
python -m runs.UrbanSound8K.PANNs.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/PANNs/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/PANNs/KD'

printf 'ALL processing is finished\n' >> $LOG_FILE