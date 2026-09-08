#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=98765
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='PAN_SC2-C_L1-05.csv'
export ANAL_FILE_L2='PAN_SC2-C_L2-05.csv'
export DATASET='SC2-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.PANNs.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 3 --fail_coll_lim 3 --max_epoch 60 \
    --lr_threshold 15 \
    --elect_weights '{"WHN":0.8, "ENQ":0.8, "END1":1.5, "END2":1.5, "ENSC":0.9, "PSH":0.8, "TST":1.5}' \
    --lrs '{"WHN":1e-5, "ENQ":1e-5, "END1":1e-6, "END2":1e-6, "ENSC":1e-5, "PSH":1e-4, "TST":1e-5}' \
    --lr_cardinalities '{"WHN":60, "ENQ":60, "END1":60, "END2":60, "ENSC":60, "PSH":60, "TST":60}' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/PANNs/TTDA' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.PANNs.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L1' --num_of_shft 4 --fail_coll_lim 4 --max_epoch 60 --lr_threshold 10 \
    --elect_weights '{"WHN":0.8, "ENQ":1.0, "END1":1.65, "END2":1.65, "ENSC":0.9, "PSH":1.0, "TST":2.1}' \
    --lrs '{"WHN":1e-5, "ENQ":1e-5, "END1":1e-6, "END2":1e-6, "ENSC":1e-5, "PSH":7e-5, "TST":1e-4}' \
    --lr_cardinalities '{"WHN":50, "ENQ":40, "END1":40, "END2":40, "ENSC":40, "PSH":60, "TST":40}' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/PANNs/TTDA' --max_mode --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m ablation_study.No_CE.SpeechCommandsV2.PANNs.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 100 --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 2.0 \
    --elect_weights '{"WHN":0.8, "ENQ":0.8, "END1":1.5, "END2":1.5, "ENSC":0.9, "PSH":0.8, "TST":1.5}' \
    --lr 1e-4 --pseudo_threshold 6.0 --lr_cardinality 80 --lr_threshold 20 \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth './result/SpeechCommandsV2/PANNs/Teach-Cons' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m ablation_study.No_CE.SpeechCommandsV2.PANNs.kd --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 160 --ctr_rt 1.0 --ctr_dist 'sq_l2' --ctr_T 1.0 \
    --elect_weights '{"WHN":0.8, "ENQ":1.0, "END1":1.65, "END2":1.65, "ENSC":0.9, "PSH":1.0, "TST":2.1}' \
    --lr 1e-4 --pseudo_threshold 7.0 --lr_cardinality 120 --lr_threshold 40 \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/PANNs/train' \
    --adpt_wght_pth './result/SpeechCommandsV2/PANNs/Teach-Cons' --seed $SEED_VAL

printf 'Student performance analysis\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.PANNs.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file $ANAL_FILE_L2 --batch_size 32 --corruption_level 'L2' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/PANNs/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/PANNs/KD' \
    --output_path './result/SpeechCommandsV2/PANNs/ablation_study/No_CE'

printf 'L1\n' >> $LOG_FILE
python -m runs.SpeechCommandsV2.PANNs.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file $ANAL_FILE_L1 --batch_size 32 --corruption_level 'L1' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/PANNs/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/PANNs/KD' \
    --output_path './result/SpeechCommandsV2/PANNs/ablation_study/No_CE'

printf 'ALL processing is finished\n' >> $LOG_FILE