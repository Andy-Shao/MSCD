#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
python -m runs.SpeechCommandsV2.HuBERT.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
    --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 30 --num_of_shft 4  \
    --fail_coll_lim 6 \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --lrs '{"WHN":7e-5, "ENQ":7e-5, "END1":7e-5, "END2":7e-5, "ENSC":1e-4, "PSH":1e-4, "TST":7e-5}' \
    --lr_gammas '{"WHN":10, "ENQ":10, "END1":10, "END2":10, "ENSC":30, "PSH":30, "TST":10}' \
    --hub_lr_decaies '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":0.55, "PSH":0.55, "TST":1.0}' \
    --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/TTDA' --wandb