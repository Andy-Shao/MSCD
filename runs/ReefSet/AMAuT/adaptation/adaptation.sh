#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.ReefSet.AMAuT.adaptation.teach_adapt --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 20 --lr '1e-4' \
    --forbid_ls 'TST' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":2.0}' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/TTDA' 