import argparse
import os
import pandas as pd
from tqdm import tqdm
import copy
from sklearn.metrics import silhouette_score
from typing import Literal
import numpy as np 
import random

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.component import Components, AudioPadding, ReduceChannel
from lib.spSet import SpeechCommandsV2C
from lib.corruption import CorruptionMeta
from ..utils import build_model
from PANNs.lib.utils import load_weight, __cal_model_path__

def silhouette_inference(args:argparse.Namespace, pan:nn.Module, clsf:nn.Module, data_loader:DataLoader) -> float:
    pan.eval(); clsf.eval()
    ttl_output, ttl_label = [], []
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            with torch.inference_mode():
                outputs = pan(features)['embedding']
                outputs = outputs.detach().cpu()
            outputs = nn.functional.normalize(outputs, p=2, dim=1)
            ttl_output.append(outputs)
            ttl_label.append(copy.deepcopy(labels))
    ttl_output = torch.concat(ttl_output, dim=0)
    ttl_label = torch.concat(ttl_label, dim=0)
    return silhouette_score(X=ttl_output.numpy(), labels=ttl_label.numpy(), metric='euclidean')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file', type=str, default='result.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--std_adpt_wght_pth', type=str)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'PANNs'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, constants.ANALYSIS)
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    data_tfs = [Components(transforms=[
        Resample(orig_freq=args.sample_rate, new_freq=constants.pann_sample_rate),
        AudioPadding(
            max_length=constants.pann_sample_rate, sample_rate=constants.pann_sample_rate, 
            random_shift=False
        ),
        ReduceChannel()
    ])] * len(corruption_types)
    records = pd.DataFrame(columns=['Model', 'Type', 'Befor Adaptation', 'After Adaptation', 'Relative Improvement'])

    print("Initialization...")
    std_pan, std_clsf = build_model(args=args, use_pre_weight=False)
    eval_sc2c = SpeechCommandsV2C(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_sc2c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    print('Embedding Analyzing...')
    print('Before Adaptation Analysis...')
    load_weight(args=args, panns=std_pan, clsf=std_clsf, mode='origin')
    org_scr = silhouette_inference(args=args, pan=std_pan, data_loader=eval_loader, clsf=std_clsf)

    print('After Adaptation Analysis...')
    load_weight(
        args=args, panns=std_pan, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION,
        metaInfo=CorruptionMeta(type=None, level=args.corruption_level)
    )
    adpt_scr = silhouette_inference(args=args, pan=std_pan, data_loader=eval_loader, clsf=std_clsf)

    records.loc[len(records)] = [args.arch, 'embedding', org_scr, adpt_scr, (adpt_scr - org_scr)/abs(org_scr)]
    records.to_csv(os.path.join(args.output_path, args.output_file))
    print('END!')