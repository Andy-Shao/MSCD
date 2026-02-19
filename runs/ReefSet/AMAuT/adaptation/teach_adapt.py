import argparse
import json
import os
import numpy as np
import random
from tqdm import tqdm
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.loss import CrossEntropyLabelSmooth
from lib.corruption import CorruptionMeta
from lib.optimizer import build_optimizer, lr_scheduler
from lib.acousSet import ReefSetC
from lib.dataset import IdxSet
from lib.component import Components, FrequenceTokenTransformer, AmplitudeToDB, OneHot2Index
from lib.component import AudioClip
from ..util import build_model, load_weight, inference

def teacher_accu_analyzing(
    args:argparse.Namespace, auts:list[nn.Module], clsfs:list[nn.Module], corruption_types:list[str],
    data_tf:nn.Module, step:int, logger
) -> None:
    print('Teacher accuracy analyzing...')
    for aut in auts: aut.eval()
    for clsf in clsfs: clsf.eval()
    accu_dic = {}
    for idx, corruption_type in tqdm(enumerate(corruption_types), total=len(corruption_types)):
        adpt_set = ReefSetC(
            root_path=args.adpt_set_path, corruption_type=corruption_type, corruption_level=args.corruption_level,
            data_tf=data_tf, label_tf=OneHot2Index()
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
            num_workers=args.num_workers
        )
        accu = inference(args=args, aut=auts[idx], clsf=clsfs[idx], data_loader=adpt_loader, tqdmable=False)
        logger.log(data={f'Adaptation/{corruption_type} Accuracy': accu}, step=step)
        accu_dic[f'{corruption_type}-{args.corruption_level}']=accu

        eval_set = ReefSetC(
            root_path=args.adpt_set_path, corruption_type=corruption_type, corruption_level=args.corruption_level,
            data_tf=data_tf, label_tf=OneHot2Index()
        )
        eval_loader = DataLoader(
            dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers
        )
        accu = inference(args=args, aut=auts[idx], clsf=clsfs[idx], data_loader=eval_loader, tqdmable=False)
        logger.log(data={f'Evaluation/{corruption_type} Accuracy': accu}, step=step)
    print({k:round(v, ndigits=4) for k,v in accu_dic.items()})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--adpt_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--elect_weights', type=str)
    ap.add_argument('--rewgt_int', type=int, default=10)
    ap.add_argument('--num_of_shft', type=int, default=3, help='maximum number of shifting teachers')
    ap.add_argument('--fail_coll_lim', type=int, default=3, help='maximum number of fail prediction be choosed in worst list')
    ap.add_argument('--max_epoch', type=int, default=20)
    ap.add_argument('--forbid_ls', type=str, default="")
    ap.add_argument('--unfrz_pos', type=int, default=-1)

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
    if not args.forbid_ls.strip():  args.forbid_ls = []
    else: args.forbid_ls = args.forbid_ls.split(',')
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, constants.TEACHER_ADAPTATION)
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TEACHER_ADAPTATION}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Teacher Adaptation', args.dataset]
    )

    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=195

    print("Initialization...")
    auts, clsfs = [], []
    max_pseudo_accu = 0.
    optimizers = []
    loss_fun = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available())
    for corruption_type in tqdm(corruption_types):
        cmeta = CorruptionMeta(type=corruption_type, level=args.corruption_level)
        aut, clsf = build_model(args=args)
        load_weight(args=args, aut=aut, clsf=clsf, mode='adaptation', metaInfo=cmeta)
        auts.append(aut)
        clsfs.append(clsf)
        optimizer = build_optimizer(lr=args.lr, auT=aut, auC=clsf, auT_decay=args.aut_lr_decay, auC_decay=args.clsf_lr_decay)
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
    # rsc_set = ReefSetC(
    #     root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
    #     data_tf=data_tfs
    # )
    # rsc_set = IdxSet(dataset=rsc_set)

    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        teacher_accu_analyzing(
            args=args, auts=auts, clsfs=clsfs, corruption_types=corruption_types, data_tf=data_tfs[0],
            step=epoch, logger=wandb_run
        )
        exit()

    wandb_run.finish()
    print('END!')