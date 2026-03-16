import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from lib.utils import ConfigDict
from AuT.lib.model import AudioClassifier

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
                outputs, _ = clsf(hub(features)[0])
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
    data_loader:DataLoader
) -> dict[str, float]:
    for hub in hubs: hub.eval()
    for clsf in clsfs: clsf.eval()
    ttl_corrs, ttl_sizes = {it: 0. for it in corruption_types}, {it: 0. for it in corruption_types}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)
            hub = hubs[i]
            clsf = clsfs[i]

            with torch.inference_mode():
                outputs, _ = clsf(hub(features)[0])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            ttl_corrs[corruption_type] += (preds==labels).sum().item()
            ttl_sizes[corruption_type] += labels.shape[0]
    accus = {}
    for corruption_type in corruption_types:
        accus[corruption_type] = ttl_corrs[corruption_type]/ttl_sizes[corruption_type]
    return accus

def build_model(args:argparse.Namespace, pre_weight:bool=True) -> tuple[torchaudio.models.Wav2Vec2Model, AudioClassifier]:
    bundle = torchaudio.pipelines.HUBERT_BASE
    if pre_weight:
        hubert = bundle.get_model().to(device=args.device)
    else:
        hubert = torchaudio.models.hubert_base().to(device=args.device)
    cfg = ConfigDict()
    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = args.class_num
    cfg.classifier['extend_size'] = 2048
    cfg.classifier['convergent_size'] = 256
    cfg.embedding = ConfigDict()
    cfg.embedding.embed_size = bundle._params['encoder_embed_dim']
    classifier = AudioClassifier(config=cfg).to(device=args.device)
    return hubert, classifier