import argparse
import numpy as np
import wandb
import json
import os
import random
from tqdm import tqdm
import copy
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.component import ReduceChannel, Components, AudioClip, OneHot2Index
from lib.corruption import CorruptionMeta
from lib.acousSet import ReefSetC
from lib.dataset import IdxSet
from ..utils import build_model
from HuBERT.lib.utils import load_weight

def pseudo_labeling(args:argparse.Namespace, corruption_types:list[str], data_tfs:list[nn.Module]):
    print('Loading all teachers...')
    teach_hubs, teach_clsfs = [], []
    for corruption_type in tqdm(corruption_types):
        teach_hub, teach_clsf = build_model(args=args)
        load_weight(
            args=args, hubert=teach_hub, clsf=teach_clsf, mode='adaptation',
            metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level),
        )
        teach_hub.eval(); teach_clsf.eval()
        teach_hubs.append(teach_hub)
        teach_clsfs.append(teach_clsf)

    print('Pseudo-labeling...')
    rfc = ReefSetC(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level, 
        data_tf=data_tfs, label_tf=OneHot2Index()
    )
    rfc = IdxSet(dataset=rfc)
    rfc_loader = DataLoader(
        dataset=rfc, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )

    for k, data in tqdm(enumerate(rfc_loader), total=len(rfc_loader)):
        idxs = data[0]
        labels = data[-1]
        for j in range(1, len(data)-1):
            features = data[j].to(args.device)
            teach_hub = teach_hubs[j-1]
            teach_clsf = teach_clsfs[j-1]
            corruption_type = corruption_types[j-1]
            with torch.inference_mode():
                outputs, _ = teach_clsf(teach_hub(features)[0])
                outputs = outputs.detach().cpu()
            preds = nn.functional.softmax(outputs, dim=1) * args.elect_weights[corruption_type]
            if j == 1: final_preds = preds
            else: final_preds = final_preds + preds
        if k == 0: 
            idx_cache = [idxs]
            pred_cache = [final_preds]
            y_t = [copy.deepcopy(labels)]
            y_s = [nn.functional.softmax(final_preds, dim=1)]
        else:
            idx_cache.append(idxs)
            pred_cache.append(final_preds)
            y_t.append(copy.deepcopy(labels))
            y_s.append(nn.functional.softmax(final_preds, dim=1))
    idx_cache = torch.concat(idx_cache, dim=0)
    pred_cache = torch.concat(pred_cache, dim=0)
    pl_roc_aud = roc_auc_score(
        y_true=torch.concat(y_t, dim=0).numpy(), y_score=torch.concat(y_s, dim=0).numpy(), average='macro',
        multi_class='ovr'
    )
    print('Pseudo-label ROC-AUC is: ', round(pl_roc_aud, ndigits=4))
    teach_hubs=None; teach_clsfs=None

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
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
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
    ap.add_argument('--interval', type=int, default=1, help='interval number')

    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
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
        config=args, tags=['Audio Classification', 'Knowledge Distillation', args.dataset]
    )

    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    data_tfs = [Components(transforms=[
        AudioClip(max_length=args.audio_length, mode='head', is_random=False),
        ReduceChannel()
    ])] * len(corruption_types)

    print("Initialization...")
    pseudo_labels = pseudo_labeling(args=args, corruption_types=corruption_types, data_tfs=data_tfs)
    print('END!')