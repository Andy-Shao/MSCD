import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.enSet import UrbanSound8KC
from lib.component import Components, AudioClip, ReduceChannel, TimeShift
from lib.dataset import MultiTFDataset
from lib.optimizer import build_optimizer, lr_scheduler
from lib.corruption import CorruptionMeta
from lib.loss import nucnm, entropy, g_entropy, mse
from ..utils import build_model, inference
from PANNs.lib.utils import load_weight, store_weight, pan_freeze

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--corruption_type', type=str, choices=['WHN', 'ENSC', 'PSH', 'TST'])
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--pan_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--mse_rate', type=float, default=1.0)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2026, help='random seed')
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--freeze_pan', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = constants.pann_sample_rate
        args.orig_sample_rate = 44100
        args.audio_length = int(4 * constants.pann_sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'PANNs'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'TTDA')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TTA_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_type}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])

    adpt_cp_us_set = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=args.corruption_type,
    )
    adpt_cp_us_set = MultiTFDataset(
        dataset=adpt_cp_us_set, 
        tfs=[Components(transforms=[
                TimeShift(shift_limit=.17, is_random=True, is_bidirection=False),
                Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
                AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                ReduceChannel()
            ]),
            Components(transforms=[
                TimeShift(shift_limit=-.17, is_random=True, is_bidirection=False),
                Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
                AudioClip(max_length=args.audio_length, mode='head', is_random=False),
                ReduceChannel()
            ]),
        ]
    )
    adpt_us_set = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=args.corruption_type,
        data_tf=Components(transforms=[
            Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            ReduceChannel()
        ])
    )
    eval_us_set = UrbanSound8KC(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=args.corruption_type,
        data_tf=Components(transforms=[
            Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            ReduceChannel()
        ])
    )
    adpt_cp_loader = DataLoader(
        dataset=adpt_cp_us_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )
    adpt_loader = DataLoader(
        dataset=adpt_us_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    eval_loader = DataLoader(
        dataset=eval_us_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    pan, clsf = build_model(args, use_pre_weight=False)
    load_weight(args=args, panns=pan, clsf=clsf, mode='origin')
    optimizer = build_optimizer(lr=args.lr, auT=pan, auC=clsf, auT_decay=args.pan_lr_decay, auC_decay=args.clsf_lr_decay)

    max_f1 = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        print('Inferencing...')
        print('Adaptation set')
        adpt_f1 = inference(args=args, pan=pan, clsf=clsf, data_loader=adpt_loader)
        print(f'F1-score is: {adpt_f1:.4f}, sample size is: {len(adpt_us_set)}')
        print('Evaluation Set')
        eval_f1 = inference(args=args, pan=pan, clsf=clsf, data_loader=eval_loader)
        print(f'F1-score is: {eval_f1:.4f}, sample size is: {len(eval_us_set)}')
        if max_f1 <= adpt_f1:
            max_f1 = adpt_f1
            store_weight(
                args=args, panns=pan, clsf=clsf, mode='adaptation', root_path=args.output_path,
                metaInfo=CorruptionMeta(type=args.corruption_type, level=args.corruption_level)
            )
        if epoch == args.max_epoch: break
        print('Adapting...')
        pan.train(); clsf.train()
        if args.freeze_pan: pan_freeze(pan=pan, batch1d=True, batch2d=True)
        ttl_size = 0.; ttl_loss = 0.; ttl_nucnm_loss = 0.
        ttl_ent_loss = 0.; ttl_gent_loss = 0.; ttl_const_loss = 0.
        for fs1, fs2, _ in tqdm(adpt_cp_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1 = clsf(pan(fs1)['embedding'])
            os2 = clsf(pan(fs2)['embedding'])

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            ent_loss = entropy(args, os1) + entropy(args, os2)
            gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os1, q=args.gent_q)
            const_loss = mse(args=args, out1=os1, out2=os2)

            loss = nucnm_loss + ent_loss + gent_loss + const_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs1.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_ent_loss += ent_loss.cpu().item()
            ttl_gent_loss += gent_loss.cpu().item()
            ttl_const_loss += const_loss.cpu().item()
        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(
                optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, 
                threshold=args.lr_threshold, momentum=args.lr_momentum
            )
        wandb_run.log(
            data={
                'Loss/ttl_loss': ttl_loss / ttl_size,
                'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
                'Loss/Entropy loss': ttl_ent_loss / ttl_size,
                'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
                'Loss/Consistency loss': ttl_const_loss / ttl_size,
                'Adaptation/F1 score': adpt_f1,
                'Adaptation/LR': learning_rate,
                'Adaptation/Max F1 score': max_f1,
                'Evaluation/F1 score': eval_f1,
            }, step=epoch, commit=True
        )
    wandb_run.finish()