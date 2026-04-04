import argparse
import os
from typing import Literal

from lib import constants
from lib.corruption import CorruptionMeta

import torch
from torch import nn

def __cal_model_path__(
    args:argparse.Namespace, mode:Literal['origin', 'adaptation', 'KD']='origin', 
    metaInfo:CorruptionMeta=None, root_path:str=None
) -> tuple[str, str]:
    if mode == 'origin':
        if root_path is None: root_path = args.orig_wght_pth
        h_p = os.path.join(root_path, f'panns-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        if root_path is None: root_path = args.adpt_wght_pth
        h_p = os.path.join(root_path, f'panns-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(root_path, f'clsf-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    elif mode == constants.STUDENT_ADAPTATION:
        if root_path is None: root_path = args.std_adpt_wght_pth
        h_p = os.path.join(root_path, f'panns-std-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(root_path, f'clsf-std-{constants.dataset_dic[args.dataset]}.pt')

    return h_p, c_p

def load_weight(
    args:argparse.Namespace, panns:nn.Module, clsf:nn.Module, 
    mode:Literal['origin', 'adaptation', 'KD']='origin', metaInfo:CorruptionMeta=None, root_path:str=None
) -> None:
    h_p, c_p = __cal_model_path__(args=args, mode=mode, metaInfo=metaInfo, root_path=root_path)
    panns.load_state_dict(state_dict=torch.load(h_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))

def store_weight(
    args:argparse.Namespace, panns:nn.Module, clsf:nn.Module, 
    mode:Literal['origin', 'adaptation', 'KD']='origin', metaInfo:CorruptionMeta=None,
    root_path:str=None
) -> None:
    a_p, c_p = __cal_model_path__(args=args, root_path=root_path, mode=mode, metaInfo=metaInfo)
    torch.save(obj=panns.state_dict(), f=a_p)
    torch.save(obj=clsf.state_dict(), f=c_p)