import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from lib.utils import ConfigDict
from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.classifier import DefClassifier

def build_model(args:argparse.Namespace, use_pre_weight:bool=True) -> tuple[Wavegram_Logmel_Cnn14, DefClassifier]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pan = Wavegram_Logmel_Cnn14(
        sample_rate=32000, 
        window_size=1024,
        hop_size=320,
        mel_bins=64, 
        fmin=50,
        fmax=14000,
        classes_num=527 #audio set
    )
    if use_pre_weight:
        ckpt = hf_hub_download(
            repo_id='nicofarr/panns_Wavegram_Logmel_Cnn14',
            filename='model.safetensors'
        )
        state = load_file(ckpt)
        # Remove 'backbone.' prefix
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        pan.load_state_dict(state_dict=state)
    cfg = ConfigDict()
    cfg.class_num = args.class_num
    cfg.embed_num = 2048
    clsf = DefClassifier(config=cfg)

    pan, clsf = pan.to(device=args.device), clsf.to(device=args.device)
    return pan, clsf

def inference(
    args:argparse.Namespace, pan:Wavegram_Logmel_Cnn14, clsf:DefClassifier, data_loader:DataLoader
) -> float:
    pan.eval(); clsf.eval()
    ttl_corr, ttl_size = 0., 0.
    for features, labels in tqdm(data_loader):
        features = features.to(args.device)

        with torch.inference_mode():
            outputs = clsf(pan(features)['embedding'])
            outputs = outputs.detach().cpu()
        _, preds = torch.max(outputs, dim=1)
        ttl_corr += (preds == labels).sum().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size