#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.ReefSet.HuBERT.train.train --dataset 'ReefSet' --dataset_root_path $BASE_PATH'/data/ReefSet_v1.0' \
    --batch_size 33 --lr '5e-4' --max_epoch 10 --model_level 'base' --use_pre_trained_weigth \
    --lr_cardinality 50 --lr_gamma 30 --lr_threshold 20 --wandb