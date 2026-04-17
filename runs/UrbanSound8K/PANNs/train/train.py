import argparse
import numpy as np
import random
import os
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse, store_model_structure_to_txt
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import CrossEntropyLabelSmooth
from lib.enSet import UrbanSound8K
from lib.component import Components, AudioPadding, ReduceChannel, Stereo2Mono, AudioClip
from ..utils import build_model, inference
from PANNs.lib.utils import __cal_model_path__, store_weight

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
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
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = constants.pann_sample_rate
        args.orig_sample_rate = 44100
        args.audio_length = int(4 * constants.pann_sample_rate)
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

    train_set = UrbanSound8K(
        root_path=args.dataset_root_path, folds=[1, 2, 3, 4, 5, 6, 7], sample_rate=args.sample_rate,
        data_tf=Components(transforms=[
            Stereo2Mono(),
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            ReduceChannel()
        ])
    )
    val_set = UrbanSound8K(
        root_path=args.dataset_root_path, folds=[8, 9, 10], sample_rate=args.sample_rate, 
        data_tf=Components(transforms=[
            Stereo2Mono(),
            AudioPadding(max_length=args.audio_length, sample_rate=args.sample_rate, random_shift=False),
            AudioClip(max_length=args.audio_length, mode='head', is_random=False),
            ReduceChannel()
        ])
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    max_f1 = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch+1}/{args.max_epoch}')
        print('Training...')
        pan.train(); clsf.train()
        # pan_freeze(pan=pan, batch1d=True, batch2d=True)
        train_loss = 0.
        for i, (features, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = clsf(pan(features)['embedding'])
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.detach().cpu(), dim=1)
            if i == 0:
                y_t = [labels.detach().cpu()]
                y_p = [preds]
            else:
                y_t.append(labels.detach().cpu())
                y_p.append(preds)
            train_loss += loss.cpu().item()
        train_f1 = f1_score(
            y_true=torch.concat(y_t, dim=0).numpy(), y_pred=torch.concat(y_p, dim=0).numpy(), 
            average='macro'
        )
        print(f'Training F1 score is: {train_f1:.4f}, sample size is: {len(train_set)}')
        y_t = None; y_p = None

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Validating...')
        val_f1 = inference(args=args, pan=pan, clsf=clsf, data_loader=val_loader)
        print(f'Validation F1 score is: {val_f1:.4f}, sample size is: {len(val_set)}')

        wandb_run.log(data={
            'Train/Loss': train_loss / len(train_loader),
            'Train/F1_score': train_f1,
            'Train/LR': learning_rate,
            'Val/F1_score': val_f1
        }, step=epoch, commit=True)

        if max_f1 <= train_f1:
            max_f1 = train_f1
            store_weight(args=args, panns=pan, clsf=clsf, mode='origin', root_path=args.output_path)
    wandb_run.finish()