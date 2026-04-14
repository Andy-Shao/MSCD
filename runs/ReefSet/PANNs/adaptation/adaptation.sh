#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Adaptation
# python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 70 --corruption_level 'L2' --corruption_type 'WHN' --lr 1e-4 --max_epoch 20 \
#     --nucnm_rate 1.2 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --wandb

# python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 70 --corruption_level 'L2' --corruption_type 'ENQ' --lr 1e-4 --max_epoch 25 \
#     --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --wandb

# python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 70 --corruption_level 'L2' --corruption_type 'END1' --lr 1e-4 --max_epoch 20 \
#     --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 --lr_momentum 0.75 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --wandb

# python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 70 --corruption_level 'L2' --corruption_type 'END2' --lr 1e-4 --max_epoch 25 \
#     --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --wandb

# python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 70 --corruption_level 'L2' --corruption_type 'ENSC' --lr 1e-4 --max_epoch 20 \
#     --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 20.6 --mse_rate 0.1 --lr_momentum 0.75 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --freeze_pan --wandb

# python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 70 --corruption_level 'L2' --corruption_type 'PSH' --lr 1e-4 --max_epoch 20 \
#     --nucnm_rate 1.0 --ent_rate 0.1 --gent_rate 0.1 --gent_q 2.1 --mse_rate 0.1 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --wandb

# python -m runs.ReefSet.PANNs.adaptation.ttda --dataset 'ReefSet' \
#     --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
#     --eval_set_path $BASE_PATH'/data/ReefSet-C' \
#     --batch_size 70 --corruption_level 'L2' --corruption_type 'TST' --lr 1e-4 --max_epoch 20 \
#     --nucnm_rate 1.0 --ent_rate 0.0 --gent_rate 0.1 --gent_q 2.1 --mse_rate 0.1 \
#     --orig_wght_pth './result/ReefSet/PANNs/train' --wandb

# Teacher Consensus
python -m runs.ReefSet.PANNs.adaptation.teach_cons --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 3 --fail_coll_lim 5 --max_epoch 20 \
    --elect_weights '{"WHN":1.0, "ENQ":0.8, "END1":1.0, "END2":0.9, "ENSC":0.8, "PSH":3.0, "TST":1.0}' \
    --lrs '{"WHN":1e-5, "ENQ":1e-4, "END1":1e-4, "END2":1e-4, "ENSC":5e-5, "PSH":1e-4, "TST":1e-4}' \
    --adpt_wght_pth './result/ReefSet/PANNs/TTDA' --wandb