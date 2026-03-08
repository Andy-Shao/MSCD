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
from lib.component import Components, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.corruption import CorruptionMeta
from lib.spSet import VocalSoundC
from lib.dataset import IdxSet, PseudoLabelSet
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import ContrastiveLoss
from ..util import build_model, load_weight, mlt_inference, store_weight

def std_accu_analyzing(
    args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, corruption_types:list[str],
    data_tfs:list[nn.Module], step:int, logger
) -> float:
    print('Adapataion set accuracy analyzing...')
    adpt_set = VocalSoundC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types, 
        data_tf=data_tfs
    )
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )
    adpt_global_accu, adpt_local_accus = mlt_inference(
        args=args, corruption_types=corruption_types, aut=aut, clsf=clsf, 
        data_loader=adpt_loader
    )
    print('Adaptation local accuracies are:', {key: round(value, ndigits=4) for key, value in adpt_local_accus.items()})
    print(f'Adaptation global accuracy is: {adpt_global_accu:.4f}')
    logger.log(data={f'Adaptation/{k} accuracy': v for k,v in adpt_local_accus.items()}, step=step)
    logger.log(data={f'Adaptation/Global accuracy': adpt_global_accu}, step=step)

    print('Evaluation set accuracy analyzing...')
    eval_set = VocalSoundC(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )
    eval_global_accu, eval_local_accus = mlt_inference(
        args=args, corruption_types=corruption_types, aut=aut, clsf=clsf,
        data_loader=eval_loader
    )
    print('Evaluation local accuracies are:', {key: round(value, ndigits=4) for key, value in eval_local_accus.items()})
    print(f'Evaluation global accuracy is: {eval_global_accu:.4f}')
    logger.log(data={f'Evaluation/{k} accuracy': v for k,v in eval_local_accus.items()}, step=step)
    logger.log(data={f'Evaluation/Global accuracy': eval_global_accu}, step=step)
    return adpt_global_accu

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
    vs_set = VocalSoundC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    vs_set = IdxSet(dataset=vs_set)
    vs_loader = DataLoader(
        dataset=vs_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )

    ttl_corr, ttl_size = 0., 0.
    idx_cache, pred_cache = [], []
    for i, data in tqdm(enumerate(vs_loader), total=len(vs_loader)):
        idxs = data[0]
        labels = data[-1]
        for j in range(1, len(data)-1):
            features = data[j].to(args.device)
            corruption_type = corruption_types[j-1]
            teach_aut, teach_clsf = teach_auts[j-1], teach_clsfs[j-1]
            with torch.inference_mode():
                outputs = teach_clsf(teach_aut(features)[1])
                outputs = outputs.detach().cpu()
            preds = nn.functional.softmax(outputs, dim=1) * args.elect_weights[corruption_type]
            if j == 1: final_preds = preds
            else: final_preds += preds
        ttl_corr += (torch.max(final_preds, dim=1)[1]==labels).sum().item()
        ttl_size += labels.shape[0]
        idx_cache.append(idxs)
        pred_cache.append(final_preds)
    idx_cache = torch.concat(idx_cache, dim=0)
    pred_cache = torch.concat(pred_cache, dim=0)
    pl_accu = ttl_corr / ttl_size
    print(f'Teacher election pseudo-labeling Accuracy is: {pl_accu:.4f}')
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
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
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
    if args.dataset == 'VocalSound':
        args.class_num = 6
        args.sample_rate = 16000
        args.audio_length = int(10 * args.sample_rate)
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
    args.n_mels=64
    n_fft=1024
    win_length=400
    hop_length=154
    mel_scale='slaney'
    args.target_length=1040

    print("Initialization...")
    data_tfs = [Components(transforms=[
        MelSpectrogram(
            sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            mel_scale=mel_scale, n_mels=args.n_mels
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer(),
    ])] * len(corruption_types)
    pseudo_labels = pseudo_labeling(args=args, corruption_types=corruption_types, data_tfs=data_tfs)
    adpt_set = VocalSoundC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
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
    max_accu = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        print('Inferencing...')
        accu = std_accu_analyzing(
            args=args, aut=std_aut, clsf=std_clsf, corruption_types=corruption_types, 
            data_tfs=data_tfs, step=epoch, logger=wandb_run
        )
        if max_accu <= accu:
            max_accu = accu
            store_weight(
                args=args, aut=std_aut, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION, root_path=args.output_path
            )
        if epoch >= args.max_epoch: break
        print('Adaptating...')
        std_aut.train(); std_clsf.train()
        # amaut_freeze(model=aut, drop=False)
        ttl_loss = 0.; ttl_clsf_loss = 0.; ttl_ctr_loss = 0.
        for adpt_data in tqdm(adpt_loader):
            labels = adpt_data[-1].to(args.device)
            for i in range(len(adpt_data)-1):
                features = adpt_data[i].to(args.device)

                outputs = std_clsf(std_aut(features)[1])

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