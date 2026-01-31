#!bin/bash
BASE_PATH=${BASE_PATH:-'/root'}

python -m runs.SpeechCommandsV2.AMAuT.noise_supression_analysis --dataset 'SpeechCommandsV2' \
    --dataset_root_path $BASE_PATH'/data/SpeechCommandsV2-C' --batch_size 32 \
    --output_file_name 'Denoiser_DNS64_analysis.csv'\
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/AMAuT/train'