#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L1.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L1-02.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L1-03.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L1-04.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L1-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_RS-C_L1_stat_anal.csv' \
    --output_path './result/ReefSet/PANNs/Analysis' \
    --report_list "${report_list}" --dataset 'RS-C' --arch 'PANNs'

report_list="\
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L2.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L2-02.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L2-03.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L2-04.csv, \
./result/ReefSet/PANNs/Analysis/PAN_RS-C_L2-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_RS-C_L2_stat_anal.csv' \
    --output_path './result/ReefSet/PANNs/Analysis' \
    --report_list "${report_list}" --dataset 'RS-C' --arch 'PANNs'