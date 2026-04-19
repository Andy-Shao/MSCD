import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os
import copy
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.corruption import CorruptionMeta
from AuT.lib.model import FCETransform, AudioClassifier
from AuT.lib.config import AuT_base

def mlt_inference(
    args:argparse.Namespace, corruption_types:list[str], aut:nn.Module, clsf:nn.Module, 
    data_loader:DataLoader
) -> tuple[float, dict[str, float]]:
    aut.eval(); clsf.eval()
    y_ts, y_ss = {k:[] for k in corruption_types}, {k:[] for k in corruption_types}
    roc_aucs = {}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            corruption_type = corruption_types[i]
            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])
                outputs = outputs.detach().cpu()
            y_ts[corruption_type].append(copy.deepcopy(labels))
            y_ss[corruption_type].append(nn.functional.softmax(outputs, dim=1))
    for i, corruption_type in enumerate(corruption_types):
        y_t = torch.concat(y_ts[corruption_type], dim=0)
        y_s = torch.concat(y_ss[corruption_type], dim=0)
        roc_aucs[corruption_type] = roc_auc_score(y_true=y_t.numpy(), y_score=y_s.numpy(), average='macro', multi_class='ovr')
        if i == 0:
            y_true = [y_t]
            y_score = [y_s]
        else:
            y_true.append(y_t)
            y_score.append(y_s)
    y_true = torch.concat(y_true, dim=0)
    y_score = torch.concat(y_score, dim=0)
    global_roc_auc = roc_auc_score(y_true=y_true.numpy(), y_score=y_score.numpy(), average='macro', multi_class='ovr')
    return global_roc_auc, roc_aucs

def teach_inference(
    args:argparse.Namespace, corruption_types:list[str], auts:list[nn.Module], clsfs:list[nn.Module],
    data_loader:DataLoader, softmax:bool=False,
) -> dict[str, float]:
    for aut in auts: aut.eval()
    for clsf in clsfs: clsf.eval()
    y_trues, y_scores = {it:[] for it in corruption_types}, {it:[] for it in corruption_types}
    roc_aucs = {}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)
            aut = auts[i]
            clsf = clsfs[i]

            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])

            y_trues[corruption_type].append(copy.deepcopy(labels))
            if softmax:
                y_scores[corruption_type].append(nn.functional.softmax(outputs.detach().cpu(), dim=1))
            else:
                y_scores[corruption_type].append(outputs.detach().cpu())
    for corruption_type in corruption_types:
        y_t = torch.concat(y_trues[corruption_type], dim=0)
        y_s = torch.concat(y_scores[corruption_type], dim=0)
        roc_aucs[corruption_type] = roc_auc_score(y_true=y_t.numpy(), y_score=y_s.numpy(), average='macro', multi_class='ovr')
        y_trues[corruption_type] = y_t
        y_scores[corruption_type] = y_s
    return roc_aucs

def build_model(args:argparse.Namespace) -> tuple[FCETransform, AudioClassifier]:
    cfg = AuT_base(class_num=args.class_num, n_mels=args.n_mels)
    cfg.embedding.in_shape = [args.n_mels, args.target_length]
    cfg.embedding.width = 128
    cfg.embedding.num_layers = [6, 8]
    cfg.embedding.embed_num = 24
    cfg.classifier.in_embed_num = 26
    aut = FCETransform(config=cfg).to(device=args.device)
    clsf = AudioClassifier(config=cfg).to(device=args.device)
    return aut, clsf

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
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode:Literal['origin', 'adaptation', 'KD']='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, mode=mode, metaInfo=metaInfo, root_path=root_path)
    aut.load_state_dict(state_dict=torch.load(a_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))

def store_weight(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, mode:Literal['origin', 'adaptation', 'KD']='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, root_path=root_path, mode=mode, metaInfo=metaInfo)
    torch.save(obj=aut.state_dict(), f=a_p)
    torch.save(obj=clsf.state_dict(), f=c_p)