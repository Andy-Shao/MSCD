import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params
from lib.corruption import corruption_meta
from lib.spSet import SpeechCommandsV2, SpeechCommandsV2C
from lib.dataset import Subset, TransferDataset
from lib.component import ReduceChannel, Components, AudioPadding, RNNoiseTransform
from .utils import build_model, load_weight
from ..utils import denormalize, normalize

def t_SNE(features: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        init='pca',
        metric='euclidean',
        random_state=42
    )
    return tsne.fit_transform(X=features.numpy())

def collect_feature(hub:nn.Module, clsf:nn.Module, data_loader:DataLoader, args:argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    hub.eval(); clsf.eval()
    out_list = []; label_list = []
    for feature, labels in tqdm(data_loader):
        feature = feature.to(args.device)
        with torch.inference_mode():
            outputs, _ = clsf(hub(feature)[0])
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        out_list.append(outputs.detach().cpu())
        label_list.append(labels.detach())
    return torch.concat(out_list, dim=0), torch.concat(label_list, dim=0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--SC2_path', type=str)
    ap.add_argument('--SC2_C_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    args = ap.parse_args()
    args.dataset = 'SpeechCommandsV2'
    args.class_num = 35
    args.sample_rate = 16000
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################
    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weight(args=args, hubert=hubert, clsf=clsf, mode='origin')
    param_no = count_ttl_params(hubert) + count_ttl_params(clsf)
        
    label_list = [7, 29, 3, 18, 34, 11, 0, 25, 14, 31]
    sc2 = Subset(
        dataset=SpeechCommandsV2(
            root_path=args.SC2_path, mode='training', download=True, data_tf=Components(transforms=[
                AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=False),
                ReduceChannel()
            ])
        ),
        label_list=label_list
    )

    sc2_loader = DataLoader(
        dataset=sc2, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    print('SC2 training set analysis')
    features, labels = collect_feature(args=args, hub=hubert, clsf=clsf, data_loader=sc2_loader)
    print(f'feature shape: {features.shape}, label size: {labels.shape}')
    X_2d = t_SNE(features=features, labels=labels)

    plt.figure(figsize=(10, 12))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.numpy(), cmap='tab10', s=8)
    plt.colorbar()
    plt.title(f't-SNE visualization: {constants.dataset_dic[args.dataset]} {constants.architecture_dic[args.arch]}')
    plt.savefig(os.path.join(args.output_path, f'tSNE_{constants.dataset_dic[args.dataset]}_{constants.architecture_dic[args.arch]}.png'))
    
    corruption_types=['END2']
    corruption_levels=['L2']

    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')
        sc2_c = Subset(
            dataset=SpeechCommandsV2C(
                root_path=args.SC2_C_path, corruption_level=cmeta.level, corruption_type=cmeta.type, 
            ),
            label_list=label_list
        )

        org_set = TransferDataset(
            dataset=sc2_c, data_tf=ReduceChannel()
        )
        org_loader = DataLoader(
            dataset=org_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
            num_workers=args.num_workers
        )
        print('SC2-C analysis')
        features, labels = collect_feature(args=args, hub=hubert, clsf=clsf, data_loader=org_loader)
        X_2d = t_SNE(features=features, labels=labels)
        plt.figure(figsize=(10, 12))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.numpy(), cmap='tab10', s=8)
        plt.colorbar()
        plt.title(f't-SNE visualization: {constants.dataset_dic[args.dataset]}-C {cmeta.type}-{cmeta.level} {constants.architecture_dic[args.arch]}')
        plt.savefig(os.path.join(args.output_path, f'tSNE_{constants.dataset_dic[args.dataset]}-C_{cmeta.type}-{cmeta.level}_{constants.architecture_dic[args.arch]}.png'))

        clean_set = TransferDataset(
            dataset=sc2_c, data_tf=Components(transforms=[
                RNNoiseTransform(sample_rate=args.sample_rate, normalize=normalize, denormalize=denormalize),
                ReduceChannel()
            ])
        )
        clean_loader = DataLoader(
            dataset=clean_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers
        )
        print('SC2-C noise suppression analysis')
        features, labels = collect_feature(args=args, hub=hubert, clsf=clsf, data_loader=clean_loader)
        X_2d = t_SNE(features=features, labels=labels)
        plt.figure(figsize=(10, 12))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.numpy(), cmap='tab10', s=8)
        plt.colorbar()
        plt.title(f't-SNE visualization: {constants.dataset_dic[args.dataset]}-C {cmeta.type}-{cmeta.level} Noise Supression {constants.architecture_dic[args.arch]}')
        plt.savefig(os.path.join(args.output_path, f'tSNE_{constants.dataset_dic[args.dataset]}-C_{cmeta.type}-{cmeta.level}_NS_{constants.architecture_dic[args.arch]}.png'))
