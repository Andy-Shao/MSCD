import argparse
from tqdm import tqdm
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.corruption import CorruptionMeta
from AuT.lib.config import AuT_base
from AuT.lib.model import FCETransform, AudioClassifier

def mlt_inference(
    args:argparse.Namespace, corruption_types:list[str], aut:nn.Module, clsf:nn.Module, 
    data_loader:DataLoader
) -> tuple[float, dict[str, float]]:
    aut.eval(); clsf.eval()
    ttl_corrs, ttl_sizes = {it: 0. for it in corruption_types}, {it: 0. for it in corruption_types}
    local_accus = {}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)

            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])
                _, preds = torch.max(outputs.detach().cpu(), dim=1)
            ttl_corrs[corruption_type] += (preds==labels).sum().item()
            ttl_sizes[corruption_type] += labels.shape[0]
    for corruption_type in corruption_types:
        local_accus[corruption_type] = ttl_corrs[corruption_type]/ttl_sizes[corruption_type]
    global_accu = sum([v for k,v in ttl_corrs.items()])/sum([v for k,v in ttl_sizes.items()])
    return global_accu, local_accus

def inference(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, data_loader:DataLoader,
    tqdmable:bool=True
):
    aut.eval(); clsf.eval()
    ttl_corr, ttl_size = 0., 0.
    if tqdmable: iterator = tqdm(data_loader)
    else: iterator = data_loader
    for features, labels in iterator:
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.inference_mode():
            outputs, _ = clsf(aut(features)[0])
        _, preds = torch.max(outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size

def build_model(args:argparse.Namespace) -> tuple[FCETransform, AudioClassifier]:
    config = AuT_base(class_num=args.class_num, n_mels=args.n_mels)
    config.embedding.in_shape = [args.n_mels, args.target_length]
    config.embedding.num_layers = [6, 12]
    config.embedding.width = 128
    config.embedding.embed_num = 13
    config.classifier.in_embed_num = 15
    clsmodel = AudioClassifier(config=config).to(device=args.device)
    auTmodel = FCETransform(config=config).to(device=args.device)

    return auTmodel, clsmodel

def __cal_model_path__(args:argparse.Namespace, mode='origin', metaInfo:CorruptionMeta=None, root_path:str=None) -> tuple[str, str]:
    assert mode in ['origin', 'adaptation'], 'No support'
    if mode == 'origin':
        if root_path is None: root_path = args.orig_wght_pth
        a_p = os.path.join(root_path, f'aut-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        if root_path is None: root_path = args.adpt_wght_path
        a_p = os.path.join(root_path, f'aut-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    return a_p, c_p

def load_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, root_path=root_path, mode=mode, metaInfo=metaInfo)
    aut.load_state_dict(state_dict=torch.load(a_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))

def store_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, root_path=root_path, mode=mode, metaInfo=metaInfo)
    torch.save(obj=aut.state_dict(), f=a_p)
    torch.save(obj=clsf.state_dict(), f=c_p)