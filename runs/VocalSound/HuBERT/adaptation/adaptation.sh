#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.VocalSound.HuBERT.adaptation.teach_cons --dataset 'VocalSound' \
    --adpt_set_path $BASE_PATH'/data/Ada-VocalSound-C' \
    --eval_set_path $BASE_PATH'/data/VocalSound-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 3 --fail_coll_lim 3 --model_level 'base' \
    --max_epoch 20 --forbid_ls 'END2' \
    --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.2, "END2":1.2, "ENSC":1.0, "PSH":1.3, "TST":1.0}' \
    --lrs '{"WHN":1e-4, "ENQ":1e-4, "END1":1e-4, "END2":1e-4, "ENSC":1e-4, "PSH":1e-4, "TST":1e-5}' \
    --adpt_wght_pth $BASE_PATH'/result/VocalSound/HuBERT/TTDA' --wandb