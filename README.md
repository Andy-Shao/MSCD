# PANS
## Software Environment
```shell
conda create --name DHAuDS python==3.13.9 -y
conda activate DHAuDS
# CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# pip install transformers==4.53.3
pip install scikit-learn==1.7.1
pip install tqdm==4.67.1
pip install pandas==2.3.1
pip install matplotlib==3.10.3
pip install jupyter==1.1.1
pip install soundfile==0.13.1
pip install wandb==0.21.0
```
