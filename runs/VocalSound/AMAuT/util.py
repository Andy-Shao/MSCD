import argparse
import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.corruption import CorruptionMeta
from AuT.lib.config import AuT_base
from AuT.lib.model import FCETransform, FCEClassifier

def partial_freeze(model:FCETransform, tf_num:int=6) -> None:
    for param in model.parameters():
        param.requires_grad = False
    
    for subm in model.layers[-tf_num:]:
        for param in subm.parameters():
            param.requires_grad = True
    for param in model.tf_norm.parameters():
        param.requires_grad = True

def teach_inference(
    args:argparse.Namespace, corruption_types:list[str], auts:list[nn.Module], clsfs:list[nn.Module],
    data_loader:DataLoader,
) -> dict[str, float]:
    for aut in auts: aut.eval()
    for clsf in clsfs: clsf.eval()
    ttl_corrs, ttl_sizes = {k:0. for k in corruption_types}, {k:0. for k in corruption_types}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            aut = auts[i]; clsf = clsfs[i]
            corruption_type = corruption_types[i]
            with torch.inference_mode():
                outputs = clsf(aut(features)[1])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            ttl_corrs[corruption_type] += (preds==labels).sum().item()
            ttl_sizes[corruption_type] += labels.shape[0]
    accus = {}
    for corruption_type in corruption_types:
        accus[corruption_type] = ttl_corrs[corruption_type]/ttl_sizes[corruption_type]
    return accus

def __cal_model_path__(args:argparse.Namespace, mode='origin', metaInfo:CorruptionMeta=None, root_path:str=None) -> tuple[str, str]:
    assert mode in ['origin', 'adaptation', constants.STUDENT_ADAPTATION], 'No support'
    if mode == 'origin':
        if root_path is None: root_path = args.orig_wght_pth
        a_p = os.path.join(root_path, f'aut-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        if root_path is None: root_path = args.adpt_wght_pth
        a_p = os.path.join(root_path, f'aut-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    elif mode == constants.STUDENT_ADAPTATION:
        if root_path is None: root_path = args.std_adpt_wght_pth
        a_p = os.path.join(root_path, f'aut-std-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(root_path, f'clsf-std-{constants.dataset_dic[args.dataset]}.pt')
    return a_p, c_p

def load_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, mode=mode, metaInfo=metaInfo, root_path=root_path)
    aut.load_state_dict(state_dict=torch.load(a_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))

def store_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, root_path=root_path, mode=mode, metaInfo=metaInfo)
    torch.save(obj=aut.state_dict(), f=a_p)
    torch.save(obj=clsf.state_dict(), f=c_p)

def build_model(args:argparse.Namespace) -> tuple[FCETransform, FCEClassifier]:
    config = AuT_base(class_num=args.class_num, n_mels=args.n_mels)
    config.embedding.in_shape = [args.n_mels, args.target_length]
    config.embedding.num_layers = [6, 4, 8]
    config.embedding.width = 64
    config.embedding.embed_num = 65
    config.classifier.in_embed_num = 2
    clsmodel = FCEClassifier(config=config).to(device=args.device)
    auTmodel = FCETransform(config=config).to(device=args.device)

    return auTmodel, clsmodel