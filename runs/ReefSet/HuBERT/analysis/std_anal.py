import argparse
import os
import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params, store_model_structure_to_txt
from lib.component import Components, AudioClip, ReduceChannel
from lib.acousSet import ReefSetC
from lib.corruption import CorruptionMeta
from ..utils import build_model, mlt_inference
from HuBERT.lib.utils import load_weight, __cal_model_path__

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='ReefSet', choices=['ReefSet'])
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file', type=str, default='result.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--std_adpt_wght_pth', type=str)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])

    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

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
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, constants.ANALYSIS)
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    data_tfs = [Components(transforms=[
        AudioClip(max_length=args.audio_length, mode='head', is_random=False),
        ReduceChannel()
    ])] * len(corruption_types)
    records = pd.DataFrame(columns=['Model', 'Num of Params', 'Corruption', 'Befor Adaptation', 'After Adaptation'])

    print("Initialization...")
    std_hub, std_clsf = build_model(args=args, pre_weight=False)
    aut_pth, clsf_pth = __cal_model_path__(
        args=args, mode=constants.STUDENT_ADAPTATION, root_path=args.output_path,
        metaInfo=CorruptionMeta(type=None, level=args.corruption_level)
    )
    aut_pth = aut_pth.replace('.pt', '.txt')
    clsf_pth = clsf_pth.replace('.pt', '.txt')
    param_num = count_ttl_params(model=std_hub) + count_ttl_params(model=std_clsf)
    store_model_structure_to_txt(model=std_hub, output_path=aut_pth)
    store_model_structure_to_txt(model=std_clsf, output_path=clsf_pth)
    eval_rsc = ReefSetC(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_rsc, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    print('Analyzing...')
    print('Before Adaptation Analysis...')
    load_weight(args=args, hubert=std_hub, clsf=std_clsf, mode='origin')
    org_glb_roc, org_lcl_rocs = mlt_inference(
        args=args, corruption_types=corruption_types, hub=std_hub, clsf=std_clsf, data_loader=eval_loader
    )

    print('After Adaptation Analysis...')
    load_weight(
        args=args, hubert=std_hub, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION,
        metaInfo=CorruptionMeta(type=None, level=args.corruption_level)
    )
    global_roc, local_rocs = mlt_inference(
        args=args, corruption_types=corruption_types, hub=std_hub, clsf=std_clsf, data_loader=eval_loader
    )
    for corruption_type, local_roc in local_rocs.items():
        records.loc[len(records)] = [
            args.arch, param_num, f'{corruption_type}-{args.corruption_level}', org_lcl_rocs[corruption_type],
            local_roc
        ]
    records.loc[len(records)] = [args.arch, param_num, 'Global', org_glb_roc, global_roc]
    records.to_csv(os.path.join(args.output_path, args.output_file))
    print('END!')