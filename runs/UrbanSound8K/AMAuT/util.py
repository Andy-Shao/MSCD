import argparse
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.corruption import CorruptionMeta
from AuT.lib.config import AuT_base
from AuT.lib.model import FCETransform, AudioClassifier

def build_model(args:argparse.Namespace) -> tuple[FCETransform, AudioClassifier]:
    cfg = AuT_base(class_num=args.class_num, n_mels=args.n_mels)
    cfg.embedding.in_shape = [args.n_mels, args.target_length]
    cfg.embedding.width = 64
    cfg.embedding.num_layers = [6, 4, 8]
    cfg.embedding.embed_num = 37
    cfg.classifier.in_embed_num = 39
    aut = FCETransform(config=cfg).to(device=args.device)
    clsf = AudioClassifier(config=cfg).to(device=args.device)
    return aut, clsf

def __cal_model_path__(args:argparse.Namespace, mode:Literal['origin', 'adaptation', 'KD']='origin', metaInfo:CorruptionMeta=None, root_path:str=None) -> tuple[str, str]:
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

def teach_inference(
    args:argparse.Namespace, corruption_types:list[str], auts:list[nn.Module], clsfs:list[nn.Module],
    data_loader:DataLoader,
) -> dict[str, float]:
    for aut in auts: aut.eval()
    for clsf in clsfs: clsf.eval()
    y_trues, y_preds = [], {k:[] for k in corruption_types}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            aut = auts[i]; clsf = clsfs[i]
            corruption_type = corruption_types[i]
            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            y_preds[corruption_type].append(preds)
        y_trues.append(labels)
    y_trues = torch.concat(y_trues, dim=0)
    f1s = {}
    for corruption_type in corruption_types:
        y_pred = torch.concat(y_preds[corruption_type], dim=0)
        f1s[corruption_type] = f1_score(y_true=y_trues.numpy(), y_pred=y_pred.numpy(), average='macro')
    return f1s

def mlt_inference(
    args:argparse.Namespace, corruption_types:list[str], aut:nn.Module, clsf:nn.Module, 
    data_loader:DataLoader
) -> tuple[float, dict[str, float]]:
    aut.eval(); clsf.eval()
    y_ts, y_ps = {k:[] for k in corruption_types}, {k:[] for k in corruption_types}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            corruption_type = corruption_types[i]
            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            y_ts[corruption_type].append(labels)
            y_ps[corruption_type].append(preds)
    local_f1s = {}
    for corruption_type in corruption_types:
        y_t = torch.concat(y_ts[corruption_type], dim=0)
        y_p = torch.concat(y_ps[corruption_type], dim=0)
        local_f1s[corruption_type] = f1_score(y_true=y_t.numpy(), y_pred=y_p.numpy(), average='macro')
        y_ts[corruption_type] = y_t
        y_ps[corruption_type] = y_p
    y_t = torch.concat([v for k,v in y_ts.items()], dim=0)
    y_p = torch.concat([v for k,v in y_ps.items()], dim=0)
    global_f1 = f1_score(y_true=y_t.numpy(), y_pred=y_p.numpy(), average='macro')
    return global_f1, local_f1s