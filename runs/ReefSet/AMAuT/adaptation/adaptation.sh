#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.ReefSet.AMAuT.adaptation.teach_adapt --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --max_epoch 20 --lr 1e-4 \
    --forbid_ls 'TST' --aut_lr_decay 0.55 --lr_gamma 30 --num_of_shft 6 --fail_coll_lim 5 \
    --elect_weights '{"WHN":1.1, "ENQ":1.0, "END1":1.2, "END2":1.1, "ENSC":1.0, "PSH":1.0, "TST":3.0}' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/AMAuT/TTDA' --wandb