#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L1.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L1-02.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L1-03.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L1-04.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L1-05.csv"

python -m runs.stat_anal --output_file_name 'AuT_SC2-C_L1_stat_anal.csv' \
    --output_path './result/SpeechCommandsV2/AMAuT/ablation_study/No_CE' \
    --report_list "${report_list}" --dataset 'SC2-C' --arch 'AMAuT'

report_list="\
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L2.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L2-02.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L2-03.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L2-04.csv, \
./result/SpeechCommandsV2/AMAuT/ablation_study/No_CE/AuT_SC2-C_L2-05.csv"

python -m runs.stat_anal --output_file_name 'AuT_SC2-C_L2_stat_anal.csv' \
    --output_path './result/SpeechCommandsV2/AMAuT/ablation_study/No_CE' \
    --report_list "${report_list}" --dataset 'SC2-C' --arch 'AMAuT'