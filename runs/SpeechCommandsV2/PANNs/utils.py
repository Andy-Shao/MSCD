import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.utils import ConfigDict
from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.classifier import PANClassifier

def build_model(args:argparse.Namespace, use_pre_weight:bool=True) -> tuple[Wavegram_Logmel_Cnn14, PANClassifier]:
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
    clsf = PANClassifier(config=cfg)

    pan, clsf = pan.to(device=args.device), clsf.to(device=args.device)
    return pan, clsf

def inference(
    args:argparse.Namespace, pan:Wavegram_Logmel_Cnn14, clsf:PANClassifier, data_loader:DataLoader
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

def teach_inference(
    args:argparse.Namespace, corruption_types:list[str], pans:list[nn.Module], clsfs:list[nn.Module],
    data_loader:DataLoader
) -> dict[str, float]:
    for pan in pans: pan.eval()
    for clsf in clsfs: clsf.eval()
    ttl_corrs, ttl_sizes = {it: 0. for it in corruption_types}, {it: 0. for it in corruption_types}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)
            pan = pans[i]
            clsf = clsfs[i]

            with torch.inference_mode():
                outputs = clsf(pan(features)['embedding'])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            ttl_corrs[corruption_type] += (preds==labels).sum().item()
            ttl_sizes[corruption_type] += labels.shape[0]
    accus = {k:ttl_corrs[k]/ttl_sizes[k] for k in corruption_types}
    return accus

def mlt_inference(
    args:argparse.Namespace, corruption_types:list[str], pan:nn.Module, clsf:nn.Module, 
    data_loader:DataLoader
) -> tuple[float, dict[str, float]]:
    pan.eval(); clsf.eval()
    ttl_corrs, ttl_sizes = {it: 0. for it in corruption_types}, {it: 0. for it in corruption_types}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)
            with torch.inference_mode():
                outputs = clsf(pan(features)['embedding'])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            ttl_corrs[corruption_type] += (preds==labels).sum().item()
            ttl_sizes[corruption_type] += labels.shape[0]
    local_accus = {k:ttl_corrs[k]/ttl_sizes[k] for k in corruption_types}
    global_accu = sum([v for k,v in ttl_corrs.items()])/sum([v for k,v in ttl_sizes.items()])
    return global_accu, local_accus