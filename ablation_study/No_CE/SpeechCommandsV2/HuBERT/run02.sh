#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=12345
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='HuB_SC2-C_L1-02.csv'
export ANAL_FILE_L2='HuB_SC2-C_L2-02.csv'
export DATASET='SC2-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.HuBERT.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 30 --num_of_shft 4  \
    --fail_coll_lim 6 --forbid_ls 'TST' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.3}' \
    --lrs '{"WHN":7e-5, "ENQ":7e-5, "END1":7e-5, "END2":7e-5, "ENSC":1e-4, "PSH":1e-4, "TST":7e-5}' \
    --lr_gammas '{"WHN":10, "ENQ":10, "END1":10, "END2":10, "ENSC":30, "PSH":30, "TST":10}' \
    --hub_lr_decaies '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":0.55, "PSH":0.55, "TST":1.0}' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.HuBERT.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 30 --num_of_shft 4  --fail_coll_lim 3 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.3, "END2":1.5, "ENSC":1.0, "PSH":1.0, "TST":2.0}' \
    --lrs '{"WHN":1e-4, "ENQ":7e-5, "END1":1e-4, "END2":1e-5, "ENSC":7e-5, "PSH":7e-5, "TST":1e-4}' \
    --lr_gammas '{"WHN":10, "ENQ":10, "END1":10, "END2":10, "ENSC":10, "PSH":10, "TST":10}' \
    --hub_lr_decaies '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/TTDA' --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m ablation_study.No_CE.SpeechCommandsV2.HuBERT.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 30 --pseudo_threshold 5.6 --ctr_rt 1.0 \
    --ctr_dist 'sq_l2' --model_level 'base' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.3}' \
    --adpt_wght_pth './result/SpeechCommandsV2/HuBERT/Teach-Cons' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m ablation_study.No_CE.SpeechCommandsV2.HuBERT.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 20 --pseudo_threshold 6.77 --ctr_rt 1.0 \
    --ctr_dist 'sq_l2' --model_level 'base' --ctr_T 1.0 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.3, "END2":1.5, "ENSC":1.0, "PSH":1.0, "TST":2.0}' \
    --adpt_wght_pth './result/SpeechCommandsV2/HuBERT/Teach-Cons' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.HuBERT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD' \
    --output_path './result/SpeechCommandsV2/HuBERT/ablation_study/No_CE'

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.HuBERT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD' \
    --output_path './result/SpeechCommandsV2/HuBERT/ablation_study/No_CE'

printf 'ALL processing is finished\n' >> $LOG_FILE