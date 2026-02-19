import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import indexes2oneHot
from lib.corruption import CorruptionMeta
from AuT.lib.model import FCETransform, AudioClassifier
from AuT.lib.config import AuT_base

def inference(
    args:argparse.Namespace, aut:FCETransform, clsf:AudioClassifier, data_loader:DataLoader,
    tqdmable:bool=True
) -> float:
    aut.eval(); clsf.eval()
    if tqdmable: iterator = tqdm(enumerate(data_loader), total=len(data_loader))
    else: iterator = enumerate(data_loader)
    for idx, (features, labels) in iterator:
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.inference_mode():
            outputs, _ = clsf(aut(features)[0])

        if idx == 0:
            y_true = indexes2oneHot(labels=labels, class_num=args.class_num)
            y_score = outputs.detach().cpu()
        else:
            y_true = torch.cat([y_true, indexes2oneHot(labels=labels, class_num=args.class_num)], dim=0)
            y_score = torch.cat([y_score, outputs.detach().cpu()], dim=0)
    val_roc_auc = roc_auc_score(y_true=y_true.numpy(), y_score=y_score.numpy(), average='macro')
    return val_roc_auc

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