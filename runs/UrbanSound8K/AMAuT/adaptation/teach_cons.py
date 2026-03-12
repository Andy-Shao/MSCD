import argparse
import os
import numpy as np
import json
import random
import wandb
from tqdm import tqdm

from lib import constants

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib.utils import print_argparse, make_unless_exits
from lib.loss import CrossEntropyLabelSmooth
from lib.corruption import CorruptionMeta
from lib.optimizer import build_optimizer, lr_scheduler
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer, AudioClip
from lib.enSet import UrbanSound8KC
from ..util import build_model, load_weight, teach_inference

def teacher_f1_analyzing(
    args:argparse.Namespace, auts:list[nn.Module], clsfs:list[nn.Module], corruption_types:list[str],
    data_tfs:list[nn.Module], step:int, logger
) -> None:
    print('Teacher accuracy analyzing...')
    print('Adaptation Set')
    adpt_set = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs
    )
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )
    adpt_f1s = teach_inference(
        args=args, corruption_types=corruption_types, auts=auts, clsfs=clsfs, data_loader=adpt_loader
    )
    print({k:round(v, ndigits=4) for k,v in adpt_f1s.items()})
    for k, v in adpt_f1s.items():
        logger.log(data={f'Adaptation/{k} F1 score': v}, step=step)

    print('Evaluation set')
    eval_set = UrbanSound8KC(
        root_path=args.eval_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )
    eval_f1s = teach_inference(
        args=args, corruption_types=corruption_types, auts=auts, clsfs=clsfs, data_loader=eval_loader
    )
    print({k:round(v, ndigits=4) for k,v in eval_f1s.items()})
    for k, v in eval_f1s.items():
        logger.log(data={f'Evaluation/{k} F1 score': v}, step=step)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--adpt_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--elect_weights', type=str)
    ap.add_argument('--num_of_shft', type=int, default=3, help='maximum number of shifting teachers')
    ap.add_argument('--fail_coll_lim', type=int, default=3, help='maximum number of fail prediction be choosed in worst list')
    ap.add_argument('--max_epoch', type=int, default=20)
    ap.add_argument('--forbid_ls', type=str, default="")
    ap.add_argument('--unfrz_pos', type=int, default=-1)

    ap.add_argument('--lr', type=float, default=1e-2)
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
    if not args.forbid_ls.strip():  args.forbid_ls = []
    else: args.forbid_ls = args.forbid_ls.split(',')
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, constants.TEACHER_CONSENSUS)
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TEACHER_CONSENSUS}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Teacher Adaptation', args.dataset]
    )

    corruption_types=['WHN', 'ENSC', 'PSH', 'TST']
    args.n_mels=64
    n_fft=2048
    win_length=800
    hop_length=300
    mel_scale='slaney'
    args.target_length=589

    print("Initialization...")
    teach_auts, teach_clsfs = [], []
    optimizers = []
    loss_fun = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available())
    for corruption_type in tqdm(corruption_types):
        cmeta = CorruptionMeta(type=corruption_type, level=args.corruption_level)
        teach_aut, teach_clsf = build_model(args=args)
        load_weight(args=args, aut=teach_aut, clsf=teach_clsf, mode='adaptation', metaInfo=cmeta)
        teach_auts.append(teach_aut)
        teach_clsfs.append(teach_clsf)
        optimizer = build_optimizer(lr=args.lr, auT=teach_aut, auC=teach_clsf, auT_decay=args.aut_lr_decay, auC_decay=args.clsf_lr_decay)
        optimizers.append(optimizer)
    data_tfs = [Components(transforms=[
        AudioClip(max_length=args.audio_length, mode='head', is_random=False),
        MelSpectrogram(
            sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=args.n_mels, mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ])] * len(corruption_types)

    max_f1 = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        teacher_f1_analyzing(
            args=args, auts=teach_auts, clsfs=teach_clsfs, corruption_types=corruption_types, 
            data_tfs=data_tfs, step=epoch, logger=wandb_run
        )
        exit()

    print('END!')