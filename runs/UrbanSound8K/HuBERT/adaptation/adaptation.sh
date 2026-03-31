#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.UrbanSound8K.HuBERT.adaptation.teach_cons --dataset 'UrbanSound8K' \
    --adpt_set_path $BASE_PATH'/data/Ada-UrbanSound8K-C' \
    --eval_set_path $BASE_PATH'/data/UrbanSound8K-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 2 --fail_coll_lim 2 --max_epoch 20 --lr '1e-5' \
    --elect_weights '{"WHN":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.0}' \
    --adpt_wght_pth $BASE_PATH'/result/UrbanSound8K/HuBERT/TTDA' --wandb