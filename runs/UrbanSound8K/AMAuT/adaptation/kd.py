import argparse
import json
import os
import wandb
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.component import AmplitudeToDB, FrequenceTokenTransformer, AudioClip, Components
from lib.corruption import CorruptionMeta
from lib.enSet import UrbanSound8KC
from lib.dataset import IdxSet
from ..util import build_model, load_weight

def pseudo_labeling(args:argparse.Namespace, corruption_types:list[str], data_tfs:list[nn.Module]):
    print('Loading all teachers...')
    teach_auts, teach_clsfs = [], []
    for corruption_type in tqdm(corruption_types):
        teach_aut, teach_clsf = build_model(args=args)
        load_weight(
            args=args, aut=teach_aut, clsf=teach_clsf, mode='adaptation',
            metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level),
        )
        teach_aut.eval(); teach_clsf.eval()
        teach_auts.append(teach_aut)
        teach_clsfs.append(teach_clsf)

    print('Pseudo-labeling...')
    us8_set = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs
    )
    us8_set = IdxSet(dataset=us8_set)
    us8_loader = DataLoader(
        dataset=us8_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )

    idx_cache, pred_cache = [], []
    y_ts, y_ps = [], []
    for i, data in tqdm(enumerate(us8_loader), total=len(us8_loader)):
        idxs = data[0]
        labels = data[-1]
        for j in range(1, len(data)-1):
            features = data[j].to(args.device)
            corruption_type = corruption_types[j-1]
            teach_aut, teach_clsf = teach_auts[j-1], teach_clsfs[j-1]
            with torch.inference_mode():
                outputs, _ = teach_clsf(teach_aut(features)[0])
                outputs = outputs.detach().cpu()
            preds = nn.functional.softmax(outputs, dim=1) * args.elect_weights[corruption_type]
            if j == 1: final_preds = preds
            else: final_preds += preds
        y_ts.append(labels)
        y_ps.append(torch.max(final_preds, dim=1)[1])
        idx_cache.append(idxs)
        pred_cache.append(final_preds)
    idx_cache = torch.concat(idx_cache, dim=0)
    pred_cache = torch.concat(pred_cache, dim=0)
    y_ts = torch.concat(y_ts, dim=0)
    y_ps = torch.concat(y_ps, dim=0)
    pl_f1 = f1_score(y_true=y_ts.numpy(), y_pred=y_ps.numpy(), average='macro')
    print(f'Teacher election pseudo-labeling F1 score is: {pl_f1:.4f}')
    teach_auts = None; teach_clsfs = None

    print('Calculating pseudo-labels...')
    pseudo_labels = {} # key -> idx, value -> smoothed label
    for i in tqdm(range(len(idx_cache)), total=len(idx_cache)):
        idx = int(idx_cache[i].item())
        pred = pred_cache[i]
        max_val, max_pos = torch.max(pred, dim=0)
        pred = torch.eye(args.class_num)[max_pos]
        if args.pseudo_threshold > max_val:
            pseudo_smooth = args.lw_def_smth
        else:
            pseudo_smooth = args.hi_def_smth
        pred = (1-pseudo_smooth)*pred + pseudo_smooth/args.class_num
        pseudo_labels[idx] = pred
    return pseudo_labels

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--elect_weights', type=str)
    ap.add_argument('--max_epoch', type=int, default=20)
    ap.add_argument('--pseudo_threshold', type=float, default=6.0)
    ap.add_argument('--hi_def_smth', type=float, default=.1)
    ap.add_argument('--lw_def_smth', type=float, default=.2)
    ap.add_argument('--ctr_rt', type=float, default=1.)
    ap.add_argument('--ctr_dist', type=str, default='l2', choices=['cos_sim', 'l2', 'sq_l2'])

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
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 44100
        args.audio_length = int(4 * args.sample_rate)
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
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.STUDENT_ADAPTATION}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Student Adaptation', args.dataset]
    )

    corruption_types=['WHN', 'ENSC', 'PSH', 'TST']
    args.n_mels=64
    n_fft=2048
    win_length=800
    hop_length=300
    mel_scale='slaney'
    args.target_length=589

    print("Initialization...")
    data_tfs = [Components(transforms=[
        AudioClip(max_length=args.audio_length, mode='head', is_random=False),
        MelSpectrogram(
            sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=args.n_mels, mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ])] * len(corruption_types)
    pseudo_labels = pseudo_labeling(args=args, corruption_types=corruption_types, data_tfs=data_tfs)

    print('END!')