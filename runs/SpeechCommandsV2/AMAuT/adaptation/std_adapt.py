import argparse
import json
import os
import numpy as np
import random
# import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse, indexes2oneHot
from lib.spSet import SpeechCommandsV2C
from lib.corruption import CorruptionMeta
from lib.dataset import IdxSet
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer
from ..utils import build_model, load_weight

def pseudo_labeling(args:argparse.Namespace, corruption_types:list[str], data_tfs:list[nn.Module]):
    print('Loading all teachers...')
    auts, clsfs = [], []
    for corruption_type in tqdm(corruption_types):
        aut, clsf = build_model(args=args)
        load_weight(
            args=args, aut=aut, clsf=clsf, mode='adaptation', 
            metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level),
        )
        aut.eval(); clsf.eval()
        auts.append(aut)
        clsfs.append(clsf)

    print('Pseudo-labeling...')
    sc2_c_set = SpeechCommandsV2C(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, 
        corruption_type=corruption_types, data_tf=data_tfs
    )
    sc2_c_set = IdxSet(dataset=sc2_c_set)
    sc2_c_loader = DataLoader(
        dataset=sc2_c_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )
    ttl_corr, ttl_size = 0., 0.
    for k, data in tqdm(enumerate(sc2_c_loader), total=len(sc2_c_loader)):
        idxs = data[0]
        labels = data[-1]
        for j in range(1, len(data)-1):
            features = data[j].to(args.device)
            aut = auts[j-1]
            clsf = clsfs[j-1]
            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])
                _, preds = torch.max(outputs.detach().cpu(), dim=1)
            preds = indexes2oneHot(labels=preds, class_num=args.class_num)
            preds = preds * args.elect_weights[corruption_type]
            if j == 1: final_preds = preds
            else: final_preds = final_preds + preds
        _, final_preds = torch.max(final_preds, dim=1)
        ttl_corr += (final_preds==labels).sum().item()
        ttl_size += labels.shape[0]
        if k == 0:
            idx_cache = [idxs]
            pred_cache = [final_preds]
        else:
            idx_cache.append(idxs)
            pred_cache.append(final_preds)
    print(f'Pseudo-labeling accuracy is: {ttl_corr/ttl_size:.4f}')

    print('Calculating pseudo-labels...')
    pseudo_labels = {} # key -> idx, value -> smoothed label

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--elect_weights', type=str)
    ap.add_argument('--max_epoch', type=int, default=20)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--aut_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--interval', type=int, default=1, help='interval number')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AMAuT'
    args.elect_weights = json.loads(args.elect_weights)
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, constants.STUDENT_ADAPTATION)
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    # wandb_run = wandb.init(
    #     project=f'{constants.PROJECT_TITLE}-{constants.STUDENT_ADAPTATION}', 
    #     name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_level}', 
    #     mode='online' if args.wandb else 'disabled', 
    #     config=args, tags=['Audio Classification', 'Student Adaptation', args.dataset]
    # )

    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104

    print("Initialization...")
    pseudo_labels = pseudo_labeling(
        args=args, corruption_types=corruption_types, 
        data_tfs=[Components(transforms=[
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                n_mels=args.n_mels, mel_scale=mel_scale
            ),
            AmplitudeToDB(top_db=80., max_out=2.),
            FrequenceTokenTransformer()
        ])] * len(corruption_types)
    )

    print('END!')