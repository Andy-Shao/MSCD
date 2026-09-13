# Multi-Shift Consensus Distillation (MSCD) for Single-Model Inference
## Software Environment
+ Docker image: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
+ GPU: RTX 4090 / RTX 5090 / RTX PRO 6000 Blackwell Workstation
```shell
conda create --name MSCD python==3.13.9 -y
conda activate MSCD
# CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torchlibrosa==0.1.0
pip install scikit-learn==1.7.1
pip install huggingface-hub==1.9.0
pip install safetensors==0.7.0
pip install tqdm==4.67.1
pip install pandas==2.3.1
pip install matplotlib==3.10.3
pip install jupyter==1.1.1
pip install soundfile==0.13.1
pip install wandb==0.21.0
```

## Processing
```shell
export BASE_PATH=${the parent directory of the project}
git clone https://github.com/Andy-Shao/DHAuDS.git
conda activate MSCD
cd MSCD
```
### Teacher Adaptation
MSCD reuses the 'CoNMix-based TTA' from [DHAuDS](https://github.com/Andy-Shao/DHAuDS). MSCD does not present the code for teacher adaptation.
If you want to process it, please run the DHAuDS project. Here we provide [HuBERT pretrained weights (tar.gz file) on SC2-C](https://drive.google.com/file/d/1Q-GA1CtpdEu-8S_pFP0cqMKzHI9OpvpY/view) (This source link includes unadapted weight files). In the default config, pretrained weights should be unzipped under 'the parent directory of the MSCD project'.

### Teacher Consensus
Here we present an example of processing teacher consensus for the HuBERT model on the SC2-C L2 level. You may need to modify a few arguments, such as 'adpt_set_path', 'eval_set_path', and 'adpt_wght_pth'. Specifically, 
+ 'adpt_set_path' is the location of the adaptation set of SC2-C. 
+ 'eval_set_path' denotes the location of the evaluation set of SC2-C.
+ 'adpt_wght_pth' is the location of the teacher adaptation-trained weights.
```shell
python -m runs.SpeechCommandsV2.HuBERT.adaptation.teach_cons --dataset 'SpeechCommandsV2' \
     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
     --batch_size 32 --corruption_level 'L2' --max_epoch 30 --num_of_shft 4  \
     --fail_coll_lim 6 --forbid_ls 'TST' \
     --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.3}' \
     --lrs '{"WHN":7e-5, "ENQ":7e-5, "END1":7e-5, "END2":7e-5, "ENSC":1e-4, "PSH":1e-4, "TST":7e-5}' \
     --lr_gammas '{"WHN":10, "ENQ":10, "END1":10, "END2":10, "ENSC":30, "PSH":30, "TST":10}' \
     --hub_lr_decaies '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":0.55, "PSH":0.55, "TST":1.0}' \
     --adpt_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/TTDA' --wandb
```
See more details in [adaptation.sh](https://github.com/Andy-Shao/MSCD/blob/main/runs/SpeechCommandsV2/HuBERT/adaptation/adaptation.sh).

### Knowledge Distillation
Here is an example of processing knowledge distillation for the HuBERT model at the SC2-C L2 level. You may need to modify a few arguments, such as 'adpt_set_path', 
'eval_set_path', 'adpt_wght_pth', and 'orig_wght_pth'. 
+ 'adpt_set_path' is the location of the adaptation set of SC2-C. 
+ 'eval_set_path' denotes the location of the evaluation set of SC2-C.
+ 'adpt_wght_pth' is the location of the teacher adaptation-trained weights.
+ 'orig_wght_pth' is the location of the unadapted weights before processing teacher adaptation. [DHAuDS](https://github.com/Andy-Shao/DHAuDS) includes the training script. For HuBERT on SC2-C, we provide pretrained weights (see the weight file link in the [Teacher Adaptation section](https://github.com/Andy-Shao/MSCD/blob/main/README.md### Teacher Consensus)).
```shell
python -m runs.SpeechCommandsV2.HuBERT.adaptation.kd --dataset 'SpeechCommandsV2' \
     --adpt_set_path $BASE_PATH'/data/Ada-SpeechCommandsV2-C' \
     --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
     --batch_size 32 --corruption_level 'L2' --max_epoch 30 --pseudo_threshold 5.6 --ctr_rt 1.0 \
     --ctr_dist 'sq_l2' --model_level 'base' \
     --elect_weights '{"WHN":1.0, "ENQ":1.0, "END1":1.0, "END2":1.0, "ENSC":1.0, "PSH":1.0, "TST":1.3}' \
     --adpt_wght_pth './result/SpeechCommandsV2/HuBERT/Teach-Cons' \
     --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' --wandb
```
See more details in [adaptation.sh](https://github.com/Andy-Shao/MSCD/blob/main/runs/SpeechCommandsV2/HuBERT/adaptation/adaptation.sh).

### Analysis
Here is an example of analyzing HuBERT performance on SC2-C at the L2 level. You may need to modify a few arguments, such as 'val_set_path', 'orig_wght_pth', and 'std_adpt_wght_pth'. 
+ 'eval_set_path' denotes the location of the evaluation set of SC2-C.
+ 'orig_wght_pth' is the unadapted weight before processing teacher adaptation. [DHAuDS](https://github.com/Andy-Shao/DHAuDS) includes the training script. For HuBERT on SC2-C, we provide pretrained weights (see the weight file link in the Teacher Adaptation section).
+ 'std_adpt_wght_pth' is the location of adapted weights after knowledge distillation. For HuBERT on SC2-C, we provide a [pretrained weight (tar.gz file)](https://drive.google.com/file/d/12CYaxX9CDkjsYyKWoIJke07f70OJs4Su/view?usp=drive_link).
```shell
python -m runs.SpeechCommandsV2.HuBERT.analysis.std_anal --dataset 'SpeechCommandsV2' \
    --eval_set_path $BASE_PATH'/data/SpeechCommandsV2-C' \
    --output_file 'HuB_SC2-C_L2.csv' --batch_size 32 --corruption_level 'L2' --model_level 'base' \
    --orig_wght_pth $BASE_PATH'/result/SpeechCommandsV2/HuBERT/train' \
    --std_adpt_wght_pth './result/SpeechCommandsV2/HuBERT/KD'
```
See more details in [analysis.sh](https://github.com/Andy-Shao/MSCD/blob/main/runs/SpeechCommandsV2/HuBERT/analysis/analysis.sh).

## Dataset
### SpeechCommands V2-C
SpeechCommandsV2-C (SC2-C) serves as a benchmark for DHAuDS in the context of test-time adaptation for audio classification. 
SC2-C, a subset of [SpeechCommands V2](https://research.google/blog/launching-the-speech-commands-dataset), utilizes only test-set samples to address domain shift. 
Each sample in SC2-C has a duration of 1 second, a sample rate of 16 kHz, and belongs to one of 35 classes. The dataset comprises two sets: adaptation and 
evaluation sets. Each set comprises seven corruption categories and two levels. Seven corruption categories are defined: WHN, ENQ, END1, END2, ENSC, TST, and PSH. 
Two levels are defined: L1 and L2, where L2 indicates a higher degree of complexity. As for the evaluation and adaptation sets, each comprises 154,070 samples. 
In total, SC2-C consists of 308,140 samples, with each category-level, such as WHN-L1 or WHN-L2, containing 11,005 samples.
+ Sample size: 308,140 (11,005 per category-level)
+ Sample rate: 16 kHz
+ Class number: 35
+ One sample length: 1s

[SC2-C Dataset Link](https://drive.google.com/drive/folders/1wBCadjjcA-n7fCAvf82uYBR6q5z_uXRm)<br/>
[Hugging Face Backup](https://huggingface.co/datasets/AndyShao90/SpeechCommandsV2-C)

### VocalSound-C
VocalSound-C (VS-C) serves as a benchmark for DHAuDS in the context of test-time adaptation for audio classification. 
VS-C, a subset of [VocalSound](https://sls.csail.mit.edu/downloads/vocalsound), utilizes only 
test-set samples to address domain shift. Each sample in VS-C has a duration of 10 seconds, a sample rate of 16 kHz, and belongs to one of 6 classes. 
The dataset comprises two sets: adaptation and evaluation sets. Each set comprises seven corruption categories and two levels. Seven corruption categories 
are defined: WHN, ENQ, END1, END2, ENSC, TST, and PSH. Two levels are defined: L1 and L2, where L2 indicates a higher degree of complexity. As for evaluation 
and adaptation sets, each comprises 50,274 samples. In total, VS-C consists of 100,548 samples, with each category-level, such as WHN-L1 or WHN-L2, containing 
3,591 samples.
+ Sample size: 100,548 (3,591 per category-level)
+ Sample rate: 16 kHz
+ One sample length: 10s
+ Class number: 6

[VS-C Dataset Link](https://drive.google.com/drive/folders/1QysFmdmFUQgQ0BlADU4eJ_xuHUxziSJX)<br/>
[Hugging Face Backup](https://huggingface.co/datasets/AndyShao90/VocalSound-C)

### UrbanSound8K-C
UrbanSound8K-C (US8-C) serves as a benchmark for DHAuDS in the context of 
test-time adaptation for audio classification. US8-C, a subset of [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), 
utilizes only test-set samples to address domain shift. Each sample in US8-C has a duration of 4 seconds, a sample rate of 44.1 kHz, 
and belongs to one of 10 classes. The dataset comprises two sets: adaptation and evaluation sets. Each set comprises four corruption categories and two levels. 
Four corruption categories are defined: WHN, ENSC, TST, and PSH. Two levels are defined: L1 and L2, where L2 indicates a higher degree of complexity. 
As for evaluation and adaptation sets, each comprises 9,836 samples. In total, US8-C consists of 19,672 samples, with each category-level, such as WHN-L1 or 
WHN-L2, containing 2,459 samples.
+ Sample size: 19,672 (2,459 per category-level)
+ Sample rate: 44.1 kHz
+ One sample length: 4s
+ Class Number: 10

[US8K-C Dataset Link](https://drive.google.com/drive/folders/1kUzBwwrRO5sIq8GUhGf8FP4HbCnb7KTh)<br/>
[Hugging face backup](https://huggingface.co/datasets/AndyShao90/UrbanSound8K-C)

### ReefSet-C
ReefSet-C (RS-C) serves as a benchmark for DHAuDS in the context of test-time adaptation for audio classification. 
RS-C, a subset of [ReefSet](https://zenodo.org/records/11071202), utilizes only test-set samples to address domain shift. Each sample in RS-C has a duration 
of 1.88 seconds, a sample rate of 16 kHz, and belongs to one of 37 classes. The dataset comprises two sets: adaptation and evaluation sets. Each set comprises 
seven corruption categories and two levels. Seven corruption categories are defined: WHN, ENQ, END1, END2, ENSC, TST, and PSH. Two levels are defined: L1 and 
L2, where L2 indicates a higher degree of complexity. As for evaluation and adaptation sets, each comprises 239,918 samples. In total, RS-C consists of 
479,836 samples, with each category-level, such as WHN-L1 or WHN-L2, containing 17,137 samples.
+ Sample size: 479,836 (17,137 per category-level)
+ Sample rate: 16 kHz
+ One sample length: 1.88s

[RS-C Dataset Link](https://drive.google.com/drive/folders/1W9GGOZTq3XSSsOlpJOQueDksHkCn3Fj4)<br/>
[Hugging Face Backup](https://huggingface.co/datasets/AndyShao90/ReefSet-C)

## Code Reference
+ [DHAuDS](https://github.com/Andy-Shao/DHAuDS)
+ [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
