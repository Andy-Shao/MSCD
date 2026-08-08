#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}
report_list="\
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L1.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L1-02.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L1-03.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L1-04.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L1-05.csv"

python -m runs.stat_anal --output_file_name 'HuB_VS-C_L1_stat_anal.csv' \
    --output_path './result/VocalSound/HuBERT/ablation_study/No_TC' \
    --report_list "${report_list}" --dataset 'VS-C' --arch 'HuBERT'

report_list="\
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L2.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L2-02.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L2-03.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L2-04.csv, \
./result/VocalSound/HuBERT/ablation_study/No_TC/HuB_VS-C_L2-05.csv"

python -m runs.stat_anal --output_file_name 'HuB_VS-C_L2_stat_anal.csv' \
    --output_path './result/VocalSound/HuBERT/ablation_study/No_TC' \
    --report_list "${report_list}" --dataset 'VS-C' --arch 'HuBERT'