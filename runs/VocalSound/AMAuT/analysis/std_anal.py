import argparse
import numpy as np
import random
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params, store_model_structure_to_txt
from lib.component import Components, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.spSet import VocalSoundC
from ..util import build_model, load_weight, __cal_model_path__, mlt_inference

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file', type=str, default='result.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--std_adpt_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])

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
    args.n_mels=64
    n_fft=1024
    win_length=400
    hop_length=154
    mel_scale='slaney'
    args.target_length=1040
    records = pd.DataFrame(columns=['Model', 'Num of Params', 'Corruption', 'Accuracy'])

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
    std_aut, std_clsf = build_model(args=args)
    load_weight(args=args, aut=std_aut, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION,)
    aut_pth, clsf_pth = __cal_model_path__(args=args, mode=constants.STUDENT_ADAPTATION, root_path=args.output_path)
    aut_pth = aut_pth.replace('.pt', '.txt')
    clsf_pth = clsf_pth.replace('.pt', '.txt')
    param_num = count_ttl_params(model=std_aut) + count_ttl_params(model=std_clsf)
    store_model_structure_to_txt(model=std_aut, output_path=aut_pth)
    store_model_structure_to_txt(model=std_clsf, output_path=clsf_pth)
    eval_set = VocalSoundC(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )

    print('Analyzing...')
    global_accu, local_accus = mlt_inference(
        args=args, corruption_types=corruption_types, aut=std_aut, clsf=std_clsf, 
        data_loader=eval_loader
    )
    for corruption_type, local_accu in local_accus.items():
        records.loc[len(records)] = [args.arch, param_num, f'{corruption_type}-{args.corruption_level}', local_accu]
    records.loc[len(records)] = [args.arch, param_num, 'Global', global_accu]
    records.to_csv(os.path.join(args.output_path, args.output_file))
    print('END!')