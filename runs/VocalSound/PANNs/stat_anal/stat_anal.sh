#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L1.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L1-02.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L1-03.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L1-04.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L1-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_VS-C_L1_stat_anal.csv' \
    --output_path './result/VocalSound/PANNs/Analysis' \
    --report_list "${report_list}" --dataset 'VS-C' --arch 'PANNs'

report_list="\
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L2.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L2-02.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L2-03.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L2-04.csv, \
./result/VocalSound/PANNs/Analysis/PAN_VS-C_L2-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_VS-C_L2_stat_anal.csv' \
    --output_path './result/VocalSound/PANNs/Analysis' \
    --report_list "${report_list}" --dataset 'VS-C' --arch 'PANNs'