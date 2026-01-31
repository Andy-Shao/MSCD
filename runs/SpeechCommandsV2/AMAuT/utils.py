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

def inference(args:argparse.Namespace, aut:FCETransform, clsf:AudioClassifier, data_loader:DataLoader):
    aut.eval(); clsf.eval()
    ttl_corr, ttl_size = 0., 0.
    for features, labels in tqdm(data_loader):
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

def load_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode='origin', metaInfo:CorruptionMeta=None
) -> None:
    assert mode in ['origin', 'adaptation'], 'No support'
    if mode == 'origin':
        a_p = os.path.join(args.orig_wght_pth, f'aut-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(args.orig_wght_pth, f'clsf-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        a_p = os.path.join(args.adpt_wght_path, f'aut-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(args.adpt_wght_path, f'clsf-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    aut.load_state_dict(state_dict=torch.load(a_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))