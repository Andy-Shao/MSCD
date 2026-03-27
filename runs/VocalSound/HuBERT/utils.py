import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.models import Wav2Vec2Model

from HuBERT.lib.model import HuBClassifier

def mlt_inference(
    args:argparse.Namespace, corruption_types:list[str], hub:nn.Module, clsf:nn.Module, 
    data_loader:DataLoader
) -> tuple[float, dict[str, float]]:
    hub.eval(); clsf.eval()
    ttl_corrs, ttl_sizes = {it: 0. for it in corruption_types}, {it: 0. for it in corruption_types}
    local_accus = {}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)
            with torch.inference_mode():
                outputs = clsf(hub(features)[0])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            ttl_corrs[corruption_type] += (preds==labels).sum().item()
            ttl_sizes[corruption_type] += labels.shape[0]
    for corruption_type in corruption_types:
        local_accus[corruption_type] = ttl_corrs[corruption_type]/ttl_sizes[corruption_type]
    global_accu = sum([v for k,v in ttl_corrs.items()])/sum([v for k,v in ttl_sizes.items()])
    return global_accu, local_accus

def teach_inference(
    args:argparse.Namespace, corruption_types:list[str], hubs:list[nn.Module], clsfs:list[nn.Module],
    data_loader:DataLoader,
) -> dict[str, float]:
    for hub in hubs: hub.eval()
    for clsf in clsfs: clsf.eval()
    ttl_corrs, ttl_size = {k:0. for k in corruption_types}, 0
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            hub = hubs[i]; clsf = clsfs[i]
            corruption_type = corruption_types[i]
            with torch.inference_mode():
                outputs = clsf(hub(features)[0])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            ttl_corrs[corruption_type] += (preds==labels).sum().item()
        ttl_size += labels.shape[0]
    accus = {}
    for corruption_type in corruption_types:
        accus[corruption_type] = ttl_corrs[corruption_type]/ttl_size
    return accus

def build_model(args:argparse.Namespace, pre_weight:bool=True) -> tuple[Wav2Vec2Model, HuBClassifier]:
    if args.model_level == 'base':
        bundle = torchaudio.pipelines.HUBERT_BASE
    elif args.model_level == 'large':
        bundle = torchaudio.pipelines.HUBERT_LARGE
    elif args.model_level == 'x-large':
        bundle = torchaudio.pipelines.HUBERT_XLARGE
    if pre_weight:
        hubert = bundle.get_model().to(device=args.device)
    else:
        hubert = torchaudio.models.hubert_base().to(device=args.device)
    classifier = HuBClassifier(
        embed_size=bundle._params['encoder_embed_dim'], class_num=args.class_num, 
        num_layers=[2, 2]
    ).to(device=args.device)
    return hubert, classifier