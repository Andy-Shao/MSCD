#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L1.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L1-02.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L1-03.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L1-04.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L1-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_SC2-C_L1_stat_anal.csv' \
    --output_path './result/SpeechCommandsV2/PANNs/ablation_study/No_CE' \
    --report_list "${report_list}" --dataset 'SC2-C' --arch 'PANNs'

report_list="\
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L2.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L2-02.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L2-03.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L2-04.csv, \
./result/SpeechCommandsV2/PANNs/ablation_study/No_CE/PAN_SC2-C_L2-05.csv"

python -m runs.stat_anal --output_file_name 'PAN_SC2-C_L2_stat_anal.csv' \
    --output_path './result/SpeechCommandsV2/PANNs/ablation_study/No_CE' \
    --report_list "${report_list}" --dataset 'SC2-C' --arch 'PANNs'