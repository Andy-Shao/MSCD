import argparse
import numpy as np
import random
import pandas as pd
import os

import torch 
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params, store_model_structure_to_txt
from lib.component import Components, AudioPadding, ReduceChannel, OneHot2Index
from lib.acousSet import ReefSetC
from ..utils import build_model, mlt_inference
from PANNs.lib.utils import __cal_model_path__, load_weight

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

    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'ReefSet':
        args.class_num = 37
        args.sample_rate = 16000
        args.audio_length = int(1.88 * constants.pann_sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'PANNs'
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
        Resample(orig_freq=args.sample_rate, new_freq=constants.pann_sample_rate),
        AudioPadding(max_length=args.audio_length, sample_rate=constants.pann_sample_rate, random_shift=False),
        ReduceChannel()
    ])] * len(corruption_types)
    records = pd.DataFrame(columns=['Model', 'Num of Params', 'Corruption', 'Befor Adaptation', 'After Adaptation'])

    print("Initialization...")
    std_pan, std_clsf = build_model(args=args, use_pre_weight=False)
    pan_pth, clsf_pth = __cal_model_path__(args=args, mode=constants.STUDENT_ADAPTATION, root_path=args.output_path)
    pan_pth = pan_pth.replace('.pt', '.txt')
    clsf_pth = clsf_pth.replace('.pt', '.txt')
    param_num = count_ttl_params(model=std_pan) + count_ttl_params(model=std_clsf)
    store_model_structure_to_txt(model=std_pan, output_path=pan_pth)
    store_model_structure_to_txt(model=std_clsf, output_path=clsf_pth)
    eval_sc2c = ReefSetC(
        root_path=args.eval_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs, label_tf=OneHot2Index()
    )
    eval_loader = DataLoader(
        dataset=eval_sc2c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    print('Analyzing...')
    print('Before Adaptation Analysis...')
    load_weight(args=args, panns=std_pan, clsf=std_clsf, mode='origin')
    org_glb_accu, org_lcl_accus = mlt_inference(
        args=args, corruption_types=corruption_types, pan=std_pan, clsf=std_clsf, data_loader=eval_loader
    )

    print('After Adaptation Analysis...')
    load_weight(args=args, panns=std_pan, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION)
    global_accu, local_accus = mlt_inference(
        args=args, corruption_types=corruption_types, pan=std_pan, clsf=std_clsf, data_loader=eval_loader
    )
    for corruption_type, local_roc in local_accus.items():
        records.loc[len(records)] = [
            args.arch, param_num, f'{corruption_type}-{args.corruption_level}', org_lcl_accus[corruption_type],
            local_roc
        ]
    records.loc[len(records)] = [args.arch, param_num, 'Global', org_glb_accu, global_accu]
    records.to_csv(os.path.join(args.output_path, args.output_file))
    print('END!')