# PANS
## Software Environment
```shell
conda create --name PANS python==3.13.9 -y
conda activate PANS
# CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install scikit-learn==1.7.1
pip install tqdm==4.67.1
pip install pandas==2.3.1
# pip install pyrnnoise==0.4.3
pip install denoiser==0.1.5
pip install matplotlib==3.10.3
pip install jupyter==1.1.1
pip install soundfile==0.13.1
pip install wandb==0.21.0
```

# Code Reference
+ [DHAuDS](https://github.com/Andy-Shao/DHAuDS)