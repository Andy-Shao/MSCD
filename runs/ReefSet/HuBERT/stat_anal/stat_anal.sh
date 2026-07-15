#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L1.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L1-02.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L1-03.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L1-04.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L1-05.csv"

python -m runs.stat_anal --output_file_name 'HuB_RS-C_L1_stat_anal.csv' \
    --output_path './result/ReefSet/HuBERT/Analysis' \
    --report_list "${report_list}" --dataset 'RS-C' --arch 'HuBERT'

report_list="\
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L2.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L2-02.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L2-03.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L2-04.csv, \
./result/ReefSet/HuBERT/Analysis/HuB_RS-C_L2-05.csv"

python -m runs.stat_anal --output_file_name 'HuB_RS-C_L2_stat_anal.csv' \
    --output_path './result/ReefSet/HuBERT/Analysis' \
    --report_list "${report_list}" --dataset 'RS-C' --arch 'HuBERT'