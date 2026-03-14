import argparse
import numpy as np
import os
import wandb
import random 
import json
from tqdm import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.corruption import CorruptionMeta
from lib.loss import CrossEntropyLabelSmooth
from lib.optimizer import build_optimizer, lr_scheduler
from lib.component import ReduceChannel
from lib.spSet import SpeechCommandsV2C
from ..utils import build_model, teach_inference
from HuBERT.lib.utils import load_weight

def teacher_accu_analyzing(
        args:argparse.Namespace, hubs:list[nn.Module], clsfs:list[nn.Module], corruption_types:list[str],
        data_tfs:list[nn.Module], step:int, logger
    ):
    print('Teacher accuracy analyzing...')
    print('Adaptation Set')
    adpt_set = SpeechCommandsV2C(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    adpt_accus = teach_inference(
        args=args, corruption_types=corruption_types, hubs=hubs, clsfs=clsfs, data_loader=adpt_loader
    )
    print({k:round(v, ndigits=4) for k,v in adpt_accus.items()})
    for corruption_type in corruption_types:
        logger.log(data={f'Adaptation/{corruption_type} Accuracy': adpt_accus[corruption_type]}, step=step)

    eval_set = SpeechCommandsV2C(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    eval_accus = teach_inference(
        args=args, corruption_types=corruption_types, hubs=hubs, clsfs=clsfs, data_loader=eval_loader
    )
    print({k:round(v, ndigits=4) for k,v in eval_accus.items()})
    for corruption_type in corruption_types:
        logger.log(data={f'Evaluation/{corruption_type} Accuracy': eval_accus[corruption_type]}, step=step)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
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

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--interval', type=int, default=1, help='interval number')

    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
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
        config=args, tags=['Audio Classification', 'Teacher Consensus', args.dataset]
    )

    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']

    print("Initialization...")
    teach_hubs, teach_clsfs = [], []
    optimizers = []
    loss_fun = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available())
    for corruption_type in tqdm(corruption_types):
        cmeta = CorruptionMeta(type=corruption_type, level=args.corruption_level)
        teach_hub, teach_clsf = build_model(args=args)
        load_weight(args=args, hubert=teach_hub, clsf=teach_clsf, mode='adaptation', metaInfo=cmeta)
        teach_hubs.append(teach_hub)
        teach_clsfs.append(teach_clsf)
        optimizer = build_optimizer(lr=args.lr, auT=teach_hub, auC=teach_clsf, auT_decay=args.hub_lr_decay, auC_decay=args.clsf_lr_decay)
        optimizers.append(optimizer)

    data_tfs = [ReduceChannel()] * len(corruption_types)

    max_pl_accu = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        teacher_accu_analyzing(
            args=args, hubs=teach_hubs, clsfs=teach_clsfs, corruption_types=corruption_types, 
            data_tfs=data_tfs, step=epoch, logger=wandb
        )
        exit()

    wandb_run.finish()
    print('END!')