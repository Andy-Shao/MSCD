import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.corruption import corruption_meta
from lib.component import ReduceChannel
from lib.dataset import TransferDataset
from lib.spSet import SpeechCommandsV2C
from noise_suppression.DNS import DNS_cache_set
from .utils import build_model, inference, load_weight
# from ..utils import normalize, denormalize

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--cache_path', type=str)

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################

    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    corruption_levels=['L1', 'L2']
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Denoiser', 'Raw_accu', 'Denoisy_accu', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    noisy_sets=[]; clean_sets=[]
    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} set generating...')
        sc2_c = SpeechCommandsV2C(
            root_path=args.dataset_root_path, corruption_level=cmeta.level, corruption_type=cmeta.type,
        )
        noisy_set = TransferDataset(
            dataset=sc2_c, data_tf=ReduceChannel()
        )
        noisy_sets.append(noisy_set)
        clean_set = TransferDataset(
            dataset=DNS_cache_set(
                dataset=sc2_c, device=args.device, sample_rate=args.sample_rate, 
                cache_path=os.path.join(args.cache_path, cmeta.type, cmeta.level), 
            ),
            data_tf=ReduceChannel()
        )
        clean_sets.append(clean_set)

    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weight(args=args, hubert=hubert, clsf=clsf, mode='origin')
    param_no = count_ttl_params(hubert) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')
        
        noisy_set = noisy_sets[idx]
        clean_set = clean_sets[idx]

        noisy_loader = DataLoader(
            dataset=noisy_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )
        clean_loader = DataLoader(
            dataset=clean_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Noisy set evaluation')
        noisy_accu = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=noisy_loader)
        print('Clean set evaluation')
        clean_accu = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=clean_loader)
        print(f'Accuracy comparision: noisy set is: {noisy_accu:.4f}, clean set is: {clean_accu:.4f}')
        records.loc[len(records)] = [args.dataset, args.arch, param_no, f'{cmeta.type}-{cmeta.level}', 'DNS64', noisy_accu, clean_accu, clean_accu-noisy_accu]
    records.to_csv(os.path.join(args.output_path, args.output_file_name))

    print('END!')