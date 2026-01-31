# PANS
## Software Environment
+ Docker image: nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
+ GPU: RTX 5090 / A100 SXM4 80GB
```shell
conda create --name PANS python==3.13.9 -y
conda activate PANS
# CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install scikit-learn==1.7.1
pip install tqdm==4.67.1
pip install pandas==2.3.1
# pip install pyrnnoise==0.4.3
pip install matplotlib==3.10.3
pip install jupyter==1.1.1
pip install soundfile==0.13.1
pip install wandb==0.21.0
```

DNS denoiser
```shell
pip install denoiser==0.1.5
```


# Code Reference
+ [DHAuDS](https://github.com/Andy-Shao/DHAuDS)