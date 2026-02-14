import argparse
import os
import json
import numpy as np
import random
from tqdm import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.corruption import CorruptionMeta
from lib.spSet import SpeechCommandsV2C
from lib.component import Components, FrequenceTokenTransformer, AmplitudeToDB
from ..utils import build_model, load_weight

def accuracy_evaluate(
    args:argparse.Namespace, teach_auts:list[nn.Module], teach_clsfs:list[nn.Module], std_aut:nn.Module,
    std_clsf:nn.Module, corruption_types:list[str], data_tfs:list[nn.Module]
) -> tuple[float, float]:
    for teach_aut in teach_auts: teach_aut.eval()
    for teach_clsf in teach_clsfs: teach_clsf.eval()
    std_aut.eval(); std_clsf.eval()
    print('Evaluation accuracy evaluation')
    sc2c_set = SpeechCommandsV2C(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    sc2c_loader = DataLoader(
        dataset=sc2c_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )
    teach_local_corrs, std_local_corrs = {it:0. for it in corruption_types}, {it:0. for it in corruption_types}
    teach_local_sizes, std_local_sizes = {it:0 for it in corruption_types}, {it:0 for it in corruption_types}
    for data in tqdm(sc2c_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            corruption_type = corruption_types[i]
            with torch.inference_mode():
                outputs, _ = std_clsf(std_aut(features)[0])
                _, preds = torch.max(outputs.detach().cpu(), dim=1)
            std_local_corrs[corruption_type] += (preds==labels).sum().item()
            std_local_sizes[corruption_type] += labels.shape[0]

            teach_aut, teach_clsf = teach_auts[i], teach_clsfs[i]
            with torch.inference_mode():
                outputs, _ = teach_clsf(teach_aut(features)[0])
                _, preds = torch.max(outputs.detach().cpu(), dim=1)
            teach_local_corrs[corruption_type] += (preds==labels).sum().item()
            teach_local_sizes[corruption_type] += labels.shape[0]
    teach_local_accus, std_local_accus = {}, {}
    for corruption_type in corruption_types:
        teach_local_accus[corruption_type] = teach_local_corrs[corruption_type]/teach_local_sizes[corruption_type]
        std_local_accus[corruption_type] = std_local_corrs[corruption_type]/std_local_sizes[corruption_type]
    teach_global_accu = sum([v for k,v in teach_local_corrs.items()])/sum([v for k,v in teach_local_sizes.items()])
    std_global_accu = sum([v for k,v in std_local_corrs.items()])/sum([v for k,v in std_local_sizes.items()])

    print(f'Student evaluation global accuracy is: {std_global_accu:.4f}')
    print('Student evaluation local accuracies are:', {k:round(v, ndigits=4) for k,v in std_local_accus.items()})
    print(f'Teacher evaluation global accuracy is: {teach_global_accu:.4f}')
    print('Teacher evaluation local accuracies are:', {k:round(v, ndigits=4) for k,v in teach_local_accus.items()})
    return teach_global_accu, std_global_accu

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--std_adpt_wght_pth', type=str)
    ap.add_argument('--adpt_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--elect_weights', type=str)
    ap.add_argument('--max_epoch', type=int, default=20)
    ap.add_argument('--pseudo_threshold', type=float, default=6.0)
    ap.add_argument('--hi_def_smth', type=float, default=.1)
    ap.add_argument('--lw_def_smth', type=float, default=.2)

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
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'AMAuT'
    args.elect_weights = json.loads(args.elect_weights)
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, constants.TEACHER_STUDENT_ADAPTATION)
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104

    print('Initialization...')
    teach_auts, teach_clsfs = [], []
    std_aut, std_clsf = build_model(args=args)
    load_weight(args=args, aut=std_aut, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION,)
    for corruption_type in tqdm(corruption_types):
        teach_aut, teach_clsf = build_model(args=args)
        load_weight(
            args=args, aut=teach_aut, clsf=teach_clsf, mode='adaptation', 
            metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level)
        )
        teach_auts.append(teach_aut)
        teach_clsfs.append(teach_clsf)

    print('Teacher-Student Adaptation')
    max_accu = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        data_tfs = [Components(transforms=[
            MelSpectrogram(
                sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                n_mels=args.n_mels, mel_scale=mel_scale
            ),
            AmplitudeToDB(top_db=80., max_out=2.),
            FrequenceTokenTransformer()
        ])] * len(corruption_types)
        print('Inferencing...')
        accuracy_evaluate(
            args=args, teach_auts=teach_auts, teach_clsfs=teach_clsfs, std_aut=std_aut, std_clsf=std_clsf,
            corruption_types=corruption_types, data_tfs=data_tfs
        )
        exit()

    print('END!')