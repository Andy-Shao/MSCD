import argparse
import json
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.spSet import SpeechCommandsV2C
from lib.corruption import CorruptionMeta
from lib.dataset import IdxSet, PseudoLabelSet
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import ContrastiveLoss
from lib.adaptation import amaut_freeze
from runs.SpeechCommandsV2.AMAuT.utils import build_model, load_weight, mlt_inference, store_weight

def clsf_rate(min_val:float, turn_epoch:int, epoch:int) -> float:
    import math
    return min_val + ((1-min_val)/(1+math.exp(epoch - turn_epoch)))

def student_accu_analyzing(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, corruption_types:list[str],
    data_tfs:list[nn.Module], step:int, logger
) -> float:
    print('Adapataion set accuracy analyzing...')
    adpt_set = SpeechCommandsV2C(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, 
        corruption_type=corruption_types, data_tf=data_tfs
    )
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    adpt_global_accu, adpt_local_accus = mlt_inference(
        args=args, corruption_types=corruption_types, aut=aut, clsf=clsf, data_loader=adpt_loader
    )
    print('Adaptation local accuracies are:', {key: round(value, ndigits=4) for key, value in adpt_local_accus.items()})
    print(f'Adaptation global accuracy is: {adpt_global_accu:.4f}')
    for k,v in adpt_local_accus.items():
        logger.log(data={f'Adaptation/{k} accuracy': v}, step=step)
    logger.log(data={f'Adaptation/Global accuracy': adpt_global_accu}, step=step)

    print('Evaluation set accuracy analyzing...')
    eval_set = SpeechCommandsV2C(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, 
        corruption_type=corruption_types, data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    global_accu, local_accus = mlt_inference(
        args=args, corruption_types=corruption_types, aut=aut, clsf=clsf, data_loader=eval_loader
    )
    print('Evaluation local accuracies are:', {key: round(value, ndigits=4) for key, value in local_accus.items()})
    print(f'Evaluation global accuracy is: {global_accu:.4f}')
    for k,v in local_accus.items():
        logger.log(data={f'Evaluation/{k} accuracy': v}, step=step)
    logger.log(data={f'Evaluation/Global accuracy': global_accu}, step=step)
    return adpt_global_accu

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
            corruption_type = corruption_types[j-1]
            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])
                outputs = outputs.detach().cpu()
            preds = nn.functional.softmax(outputs, dim=1) * args.elect_weights[corruption_type]
            if j == 1: final_preds = preds
            else: final_preds = final_preds + preds
        ttl_corr += (torch.max(final_preds, dim=1)[1]==labels).sum().item()
        ttl_size += labels.shape[0]
        if k == 0:
            idx_cache = [idxs]
            pred_cache = [final_preds]
        else:
            idx_cache.append(idxs)
            pred_cache.append(final_preds)
    idx_cache = torch.concat(idx_cache, dim=0)
    pred_cache = torch.concat(pred_cache, dim=0)
    print(f'Pseudo-labeling accuracy is: {ttl_corr/ttl_size:.4f}')
    auts=None; clsfs=None

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
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
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
    ap.add_argument('--ctr_T', type=float, default=1.)

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
    adpt_set = SpeechCommandsV2C(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, 
        corruption_type=corruption_types, 
        data_tf=[Components(transforms=[
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                n_mels=args.n_mels, mel_scale=mel_scale
            ),
            AmplitudeToDB(top_db=80., max_out=2.),
            FrequenceTokenTransformer()
        ])] * len(corruption_types)
    )
    adpt_set = PseudoLabelSet(dataset=adpt_set, pseudo_labels=pseudo_labels, label_position=7)
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, 
        num_workers=args.num_workers
    )
    aut, clsf = build_model(args=args)
    load_weight(args=args, aut=aut, clsf=clsf, mode='origin')
    optimizer = build_optimizer(
        lr=args.lr, auT=aut, auC=clsf, auT_decay=args.aut_lr_decay, auC_decay=args.clsf_lr_decay
    )
    ctr_loss_fun = ContrastiveLoss(
        hi_def_smth=args.hi_def_smth, class_num=args.class_num, device=args.device, dist=args.ctr_dist,
        tau=args.ctr_T
    )

    print('Student Adaptation')
    max_accu = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        print('Inferencing...')
        accu = student_accu_analyzing(
            args=args, aut=aut, clsf=clsf, corruption_types=corruption_types, step=epoch,
            data_tfs=[Components(transforms=[
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    n_mels=args.n_mels, mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])] * len(corruption_types), logger=wandb_run
        )
        if max_accu <= accu:
            max_accu = accu
            store_weight(
                args=args, aut=aut, clsf=clsf, mode=constants.STUDENT_ADAPTATION, root_path=args.output_path,
                metaInfo=CorruptionMeta(type=None, level=args.corruption_level)
            )
        
        if epoch >= args.max_epoch: break
        print('Adaptating...')
        aut.train(); clsf.train()
        # amaut_freeze(model=aut, drop=False) # Do not use the Batch calibration
        ttl_loss = 0.; ttl_clsf_loss = 0.; ttl_ctr_loss = 0.
        for adpt_data in tqdm(adpt_loader):
            labels = adpt_data[-1].to(args.device)
            for i in range(len(adpt_data)-1):
                features = adpt_data[i].to(args.device)

                outputs, _ = clsf(aut(features)[0])

                # classification loss
                clsf_loss = (-labels * nn.functional.log_softmax(outputs, dim=1)).sum(dim=1) # cross-entropy loss
                clsf_loss = clsf_loss.mean()

                # contrastive loss
                if args.ctr_rt > 0.:
                    ctr_loss = args.ctr_rt * ctr_loss_fun(outputs, labels)
                else: ctr_loss = torch.tensor(0.).to(device=args.device)

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