#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=98765
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='HuB_VS-C_L1-05.csv'
export ANAL_FILE_L2='HuB_VS-C_L2-05.csv'
export DATASET='VS-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.adaptation.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --model_level 'base' --max_epoch 20 --lr 1e-4 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.2, "END2":1.2, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 5.69 \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.adaptation.kd --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L1' --model_level 'base' --max_epoch 20 --lr 1e-4 \
    --elect_weights '{"WHN":1.2, "ENQ":1.0, "END1":1.3, "END2":1.3, "ENSC":1.0, "PSH":1.2, "TST":1.0}' \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 6.15 \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/TTDA' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --std_adpt_wght_pth './result/VocalSound/HuBERT/KD' \
    --output_path './result/VocalSound/HuBERT/ablation_study/No_TC'

printf 'L1\n' >> $LOG_FILE
python -m runs.VocalSound.HuBERT.analysis.std_anal --dataset 'VocalSound' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/train' \
    --std_adpt_wght_pth './result/VocalSound/HuBERT/KD' \
    --output_path './result/VocalSound/HuBERT/ablation_study/No_TC'

printf 'ALL processing is finished\n' >> $LOG_FILE