# Multi-Shift Consensus Distillation (MSCD): Robust Test-Time Adaptation via Elected Pseudo-Labels
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

## Code Reference
+ [DHAuDS](https://github.com/Andy-Shao/DHAuDS)
+ [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
