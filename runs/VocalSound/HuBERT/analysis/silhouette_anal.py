import argparse
import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import silhouette_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.component import ReduceChannel
from lib.spSet import VocalSoundC
from ..utils import build_model
from HuBERT.lib.utils import load_weight

def silhouette_inference(args:argparse.Namespace, hub:nn.Module, clsf:nn.Module, data_loader:DataLoader) -> float:
    hub.eval(); clsf.eval()
    ttl_output, ttl_label = [], []
    for data in tqdm(data_loader):
        labels = data[-1].detach()
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            with torch.inference_mode():
                outputs = hub(features)[0]
                outputs = outputs.detach().cpu()
            outputs = torch.mean(outputs, dim=1)
            outputs = nn.functional.normalize(outputs, p=2, dim=1)
            ttl_output.append(outputs)
            ttl_label.append(labels)
    ttl_output = torch.concat(ttl_output, dim=0)
    ttl_label = torch.concat(ttl_label, dim=0)
    return silhouette_score(X=ttl_output.numpy(), labels=ttl_label.numpy(), metric='euclidean')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file', type=str, default='result.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--std_adpt_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])

    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
        args.audio_length = int(10 * args.sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
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
    data_tfs = [ReduceChannel()] * len(corruption_types)
    records = pd.DataFrame(columns=['Model', 'Type', 'Befor Adaptation', 'After Adaptation', 'Relative Improvement'])

    print("Initialization...")
    std_hub, std_clsf = build_model(args=args, pre_weight=False)
    eval_vsc = VocalSoundC(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_vsc, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    print('Embedding Analyzing...')
    print('Before Adaptation Analysis...')
    load_weight(args=args, hubert=std_hub, clsf=std_clsf, mode='origin')
    org_scr = silhouette_inference(args=args, hub=std_hub, clsf=std_clsf, data_loader=eval_loader)

    print('After Adaptation Analysis...')
    load_weight(args=args, hubert=std_hub, clsf=std_clsf, mode='KD')
    adpt_scr = silhouette_inference(args=args, hub=std_hub, clsf=std_clsf, data_loader=eval_loader)

    records.loc[len(records)] = [args.arch, 'embedding', org_scr, adpt_scr, (adpt_scr - org_scr)/abs(org_scr)]
    records.to_csv(os.path.join(args.output_path, args.output_file))
    print('END!')