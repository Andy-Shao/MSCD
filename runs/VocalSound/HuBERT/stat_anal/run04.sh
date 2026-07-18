#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=56789
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='HuB_VS-C_L1-04.csv'
export ANAL_FILE_L2='HuB_VS-C_L2-04.csv'
export DATASET='VS-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 4 --fail_coll_lim 3 --model_level 'base' \
    --max_epoch 15  --forbid_ls 'WHN' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.2, "END2":1.2, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --lrs '{"WHN":1e-5, "ENQ":1e-4, "END1":1e-5, "END2":1e-5, "ENSC":1e-4, "PSH":1e-5, "TST":1e-5}' \
    --lr_momentums '{"WHN":0.9, "ENQ":0.9, "END1":0.9, "END2":0.9, "ENSC":0.9, "PSH":0.9, "TST":0.75}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L1' --num_of_shft 2 --fail_coll_lim 3 --model_level 'base' \
    --max_epoch 15 --forbid_ls 'END1,END2' \
    --elect_weights '{"WHN":1.2, "ENQ":1.0, "END1":1.3, "END2":1.3, "ENSC":1.0, "PSH":1.2, "TST":1.0}' \
    --lrs '{"WHN":1e-4, "ENQ":1e-4, "END1":1e-5, "END2":1e-5, "ENSC":1e-4, "PSH":1e-5, "TST":1e-4}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/TTDA' --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.adaptation.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --model_level 'base' --max_epoch 20 --lr 1e-4 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.2, "END2":1.2, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 5.69 \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --adpt_wght_pth './result/VocalSound/HuBERT/Teach-Cons' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.adaptation.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L1' --model_level 'base' --max_epoch 20 --lr 1e-4 \
    --elect_weights '{"WHN":1.2, "ENQ":1.0, "END1":1.3, "END2":1.3, "ENSC":1.0, "PSH":1.2, "TST":1.0}' \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 6.15 \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --adpt_wght_pth './result/VocalSound/HuBERT/Teach-Cons' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --std_adpt_wght_pth './result/VocalSound/HuBERT/KD'

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --std_adpt_wght_pth './result/VocalSound/HuBERT/KD'

printf 'ALL processing is finished\n' >> $LOG_FILE