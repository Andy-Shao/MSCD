#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
export SEED_VAL=2026
export LOG_FILE=$BASE_PATH'/MSCD.log'
export ANAL_FILE_L1='AuT_US8-C_L1.csv'
export ANAL_FILE_L2='AuT_US8-C_L2.csv'
export DATASET='US8-C'

> $LOG_FILE
printf 'Processing Log\n' >> $LOG_FILE
printf 'TTA on '$DATASET' , seed is:'$SEED_VAL'\n' >> $LOG_FILE
printf '================================\n' >> $LOG_FILE

printf 'Teacher Consensus\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m runs.UrbanSound8K.AMAuT.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 3 --fail_coll_lim 2 --max_epoch 30 \
    --lr '1e-5' --lr_cardinality 60 \
    --elect_weights '{"WHN":0.9, "ENSC":1.0, "PSH":0.9, "TST":1.0}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/TTDA' --freeze_aut --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m runs.UrbanSound8K.AMAuT.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L1' --num_of_shft 3 --fail_coll_lim 2 --max_epoch 30 \
    --lr '1e-5' --lr_cardinality 60 \
    --elect_weights '{"WHN":1.0, "ENSC":1.0, "PSH":0.9, "TST":1.0}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/TTDA' --freeze_aut --seed $SEED_VAL

printf 'Knowledge Distillation\n' >> $LOG_FILE
printf 'L2\n' >> $LOG_FILE
python -m ablation_study.No_CE.UrbanSound8K.AMAuT.kd --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 20 --lr '1e-4' \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 2.92 \
    --elect_weights '{"WHN":0.9, "ENSC":1.0, "PSH":0.9, "TST":1.0}' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/train' \
    --adpt_wght_pth './result/UrbanSound8K/AMAuT/Teach-Cons' --seed $SEED_VAL

printf 'L1\n' >> $LOG_FILE
python -m ablation_study.No_CE.UrbanSound8K.AMAuT.kd --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L1' --max_epoch 20 --lr '1e-4' \
    --ctr_rt 1.0 --ctr_dist 'sq_l2' --pseudo_threshold 3.0 \
    --elect_weights '{"WHN":1.0, "ENSC":1.0, "PSH":0.9, "TST":1.0}' \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/train' \
    --adpt_wght_pth './result/UrbanSound8K/AMAuT/Teach-Cons' --seed $SEED_VAL

printf 'Student Analysis\n' >> $LOG_FILE
python -m runs.UrbanSound8K.AMAuT.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file $ANAL_FILE_L2 --corruption_level 'L2' --batch_size 32 \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/AMAuT/KD' \
    --output_path './result/UrbanSound8K/AMAuT/ablation_study/No_CE'

python -m runs.UrbanSound8K.AMAuT.analysis.std_anal --dataset 'UrbanSound8K' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --output_file $ANAL_FILE_L1 --corruption_level 'L1' --batch_size 32 \
    --orig_wght_pth $BASE_PATH'/result/UrbanSound8K/AMAuT/train' \
    --std_adpt_wght_pth './result/UrbanSound8K/AMAuT/KD' \
    --output_path './result/UrbanSound8K/AMAuT/ablation_study/No_CE'

printf 'ALL processing is finished\n' >> $LOG_FILE