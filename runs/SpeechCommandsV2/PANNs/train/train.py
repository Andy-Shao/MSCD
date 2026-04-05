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
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt
from lib.spSet import SpeechCommandsV2
from lib.component import Components, ReduceChannel, AudioPadding
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import CrossEntropyLabelSmooth
from PANNs.lib.utils import __cal_model_path__, store_weight, pan_freeze
from ..utils import build_model, inference

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--pan_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--max_epoch', type=int, default=30)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default='2026')
    ap.add_argument('--smooth', type=float, default=.1)

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.arch = 'PANNs'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'train')

    torch.backends.cudnn.benchmark == True
    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    print_argparse(args=args)
    ############################################################

    make_unless_exits(args.output_path)
    make_unless_exits(args.dataset_root_path)

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )
    pan, clsf = build_model(args, use_pre_weight=True)
    optimizer = build_optimizer(lr=args.lr, auT=pan, auC=clsf, auT_decay=args.pan_lr_decay, auC_decay=args.clsf_lr_decay)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, reduction=True, use_gpu=torch.cuda.is_available(), epsilon=.1)

    pan_pth, clsf_pth = __cal_model_path__(args=args, mode='origin', root_path=args.output_path)
    pan_pth, clsf_pth = pan_pth.replace('.pt', '.txt'), clsf_pth.replace('.pt', '.txt')
    store_model_structure_to_txt(model=pan, output_path=pan_pth)
    store_model_structure_to_txt(model=clsf, output_path=clsf_pth)

    train_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='training', download=True, 
        data_tf=Components(transforms=[
            Resample(orig_freq=args.sample_rate, new_freq=constants.pann_sample_rate),
            AudioPadding(max_length=constants.pann_sample_rate, sample_rate=constants.pann_sample_rate, random_shift=False),
            ReduceChannel()
        ])
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, 
        num_workers=args.num_workers
    )
    val_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='validation', download=False,
        data_tf=Components(transforms=[
            Resample(orig_freq=args.sample_rate, new_freq=constants.pann_sample_rate),
            AudioPadding(max_length=constants.pann_sample_rate, sample_rate=constants.pann_sample_rate, random_shift=False),
            ReduceChannel()
        ])
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        pan.train(); clsf.train()
        pan_freeze(pan=pan, batch1d=True, batch2d=True)
        train_loss = 0.
        ttl_corr, ttl_size = 0., 0.
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = clsf(pan(features)['embedding'])
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
            train_loss += loss.detach().cpu().item()
        train_accu = ttl_corr / ttl_size
        print(f'Training accuracy is: {train_accu:.4f}, sample size is: {len(train_set)}')

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(
                optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, 
                threshold=args.lr_threshold
            )

        print('Validating...')
        val_accu = inference(args=args, pan=pan, clsf=clsf, data_loader=val_loader)
        print(f'Validation accuracy is: {val_accu:.4f}, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/Accuracy': train_accu,
            'Train/LR': learning_rate,
            'Val/Accuracy': val_accu
        }, step=epoch, commit=True)

        if max_accu <= train_accu:
            max_accu = train_accu
            store_weight(args=args, panns=pan, clsf=clsf, mode='origin', root_path=args.output_path)
    wandb_run.finish()
    print('END!')