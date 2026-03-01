import argparse
import os
import numpy as np
import json
import wandb
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer, AudioClip
from lib.component import OneHot2Index
from lib.corruption import CorruptionMeta
from lib.acousSet import ReefSetC
from lib.dataset import IdxSet, PseudoLabelSet
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import ContrastiveLoss
from ..util import load_weight, build_model, mlt_inference, store_weight

def std_roc_auc_analyzing(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, corruption_types:list[str],
    data_tfs:list[nn.Module], step:int, logger
) -> float:
    print('Adapataion set accuracy analyzing...')
    adpt_set = ReefSetC(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs, label_tf=OneHot2Index()
    )
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    adpt_global_roc_auc, adpt_roc_aucs = mlt_inference(
        args=args, corruption_types=corruption_types, aut=aut, clsf=clsf, data_loader=adpt_loader
    )
    print('Adptation local ROC-AUC are:', {k:round(v, ndigits=4) for k,v in adpt_roc_aucs.items()})
    print(f'Adptation global ROC-AUC is: {adpt_global_roc_auc:.4f}')
    for k,v in adpt_roc_aucs.items():
        logger.log(data={f'Adaptation/{k} ROC-AUC': v}, step=step)
    logger.log(data={f'Adaptation/Global ROC-AUC': adpt_global_roc_auc}, step=step)

    print('Evaluation set accuracy analyzing...')
    eval_set = ReefSetC(
        root_path=args.eval_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs, label_tf=OneHot2Index()
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    eval_global_roc_auc, eval_roc_aucs = mlt_inference(
        args=args, corruption_types=corruption_types, aut=aut, clsf=clsf, data_loader=eval_loader
    )
    print('Evaluation local ROC-AUC are:', {k:round(v, ndigits=4) for k,v in eval_roc_aucs.items()})
    print(f'Evaluation global ROC-AUC is: {eval_global_roc_auc:.4f}')
    for k,v in eval_roc_aucs.items():
        logger.log(data={f'Evaluation/{k} ROC-AUC': v}, step=step)
    logger.log(data={f'Evaluation/Global ROC-AUC': eval_global_roc_auc}, step=step)
    return adpt_global_roc_auc

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
    rsc_set = ReefSetC(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs, label_tf=OneHot2Index()
    )
    rsc_set = IdxSet(dataset=rsc_set)
    rsc_loader = DataLoader(
        dataset=rsc_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    y_t, y_s = [], []
    for i, data in tqdm(enumerate(rsc_loader), total=len(rsc_loader)):
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
        if i == 0:
            y_t = [copy.deepcopy(labels)]
            y_s = [nn.functional.softmax(final_preds, dim=1)]
            idx_cache = [idxs]
            pred_cache = [final_preds]
        else:
            y_t.append(copy.deepcopy(labels))
            y_s.append(nn.functional.softmax(final_preds, dim=1))
            idx_cache.append(idxs)
            pred_cache.append(final_preds)
    idx_cache = torch.concat(idx_cache, dim=0)
    pred_cache = torch.concat(pred_cache, dim=0)
    y_t = torch.concat(y_t, dim=0)
    y_s = torch.concat(y_s, dim=0)
    pl_roc_auc = roc_auc_score(y_true=y_t.numpy(), y_score=y_s.numpy(), average='macro', multi_class='ovr')
    print(f'Teacher election pseudo-labeling ROC-AUC is: {pl_roc_auc:.4f}')
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
    ap.add_argument('--aut_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--interval', type=int, default=1, help='interval number')

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

    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=195

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
    adpt_set = ReefSetC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs, label_tf=OneHot2Index()
    )
    adpt_set = PseudoLabelSet(dataset=adpt_set, pseudo_labels=pseudo_labels, label_position=len(corruption_types))
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, 
        num_workers=args.num_workers
    )
    std_aut, std_clsf = build_model(args=args)
    load_weight(args=args, aut=std_aut, clsf=std_clsf, mode='origin',)
    optimizer = build_optimizer(
        lr=args.lr, auT=std_aut, auC=std_clsf, auT_decay=args.aut_lr_decay, 
        auC_decay=args.clsf_lr_decay
    )
    ctr_loss_fun = ContrastiveLoss(
        hi_def_smth=args.hi_def_smth, class_num=args.class_num, device=args.device, 
        dist=args.ctr_dist
    )

    print('Student Adaptation')
    max_roc_auc = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        print('Inferencing...')
        std_roc_auc = std_roc_auc_analyzing(
            args=args, aut=std_aut, clsf=std_clsf, corruption_types=corruption_types,
            data_tfs=data_tfs, step=epoch, logger=wandb_run
        )
        if max_roc_auc <= std_roc_auc:
            max_roc_auc = std_roc_auc
            store_weight(
                args=args, aut=std_aut, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION, 
                root_path=args.output_path
            )
        
        if epoch >= args.max_epoch: break
        print('Adaptating...')
        std_aut.train(); std_clsf.train()
        ttl_loss = 0.; ttl_clsf_loss = 0.; ttl_ctr_loss = 0.
        for adpt_data in tqdm(adpt_loader):
            labels = adpt_data[-1].to(args.device)
            for i in range(len(adpt_data)-1):
                features = adpt_data[i].to(args.device)

                outputs, _ = std_clsf(std_aut(features)[0])

                # classification loss
                clsf_loss = (-labels * nn.functional.log_softmax(outputs, dim=1)).sum(dim=1) # cross-entropy loss
                clsf_loss = clsf_loss.mean()

                # contrastive loss
                if args.ctr_rt > 0.:
                    ctr_loss = ctr_loss_fun(outputs, labels)
                else: ctr_loss = torch.tensor(0.).to(args.device)

                if i == 0:
                    loss = clsf_loss + ctr_loss
                else: 
                    loss += clsf_loss + ctr_loss
                ttl_clsf_loss += clsf_loss.detach().cpu().item()
                ttl_ctr_loss += ctr_loss.detach().cpu().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ttl_loss += loss.detach().cpu().item()
        wandb_run.log(data={
            'Loss/TTL Loss': ttl_loss/(len(adpt_loader)*len(corruption_types)),
            'Loss/Classification loss': ttl_clsf_loss/(len(adpt_loader)*len(corruption_types)),
            'Loss/Contrastive loss': ttl_ctr_loss/(len(adpt_loader)*len(corruption_types)),
        }, step=epoch)
        if epoch % args.interval == 0:
            lr_scheduler(
                optimizer=optimizer, epoch=epoch+1, lr_cardinality=args.lr_cardinality,
                gamma=args.lr_gamma, threshold=args.lr_threshold, momentum=args.lr_momentum
            )
    wandb_run.finish()
    print('END!')