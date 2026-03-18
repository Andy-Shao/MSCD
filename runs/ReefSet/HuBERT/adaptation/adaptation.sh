#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

# Teacher Consensus
python -m runs.ReefSet.HuBERT.adaptation.teach_cons --dataset 'ReefSet' \
    --adpt_set_path $BASE_PATH'/data/Ada-ReefSet-C' \
    --eval_set_path $BASE_PATH'/data/ReefSet-C' \
    --batch_size 32 --corruption_level 'L2' --num_of_shft 3 --fail_coll_lim 3 --max_epoch 2 \
    --lr 1e-4 \
    --elect_weights '{"WHN":1.0, "ENQ":1.2, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":2.0, "TST":1.5}' \
    --adpt_wght_pth $BASE_PATH'/result/ReefSet/HuBERT/TTDA'