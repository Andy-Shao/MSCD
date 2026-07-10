#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L1.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L1-02.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L1-03.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L1-04.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L1-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_US8-C_L1_stat_anal.csv' \
    --output_path './result/UrbanSound8K/PANNs/Analysis' \
    --report_list "${report_list}" --dataset 'US8-C' --arch 'PANNs'

report_list="\
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L2.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L2-02.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L2-03.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L2-04.csv, \
./result/UrbanSound8K/PANNs/Analysis/PAN_US8-C_L2-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_US8-C_L2_stat_anal.csv' \
    --output_path './result/UrbanSound8K/PANNs/Analysis' \
    --report_list "${report_list}" --dataset 'US8-C' --arch 'PANNs'