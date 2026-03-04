#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.VocalSound.adaptation.teach_adapt --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --rewgt_int -1 --num_of_shft 3 --fail_coll_lim 3 \
    --max_epoch 20 --lr '1e-4' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/AMAuT/TTDA'