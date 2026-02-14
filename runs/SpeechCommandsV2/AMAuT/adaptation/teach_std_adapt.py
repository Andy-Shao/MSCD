import argparse
import os
import json
import numpy as np
import random
from tqdm import tqdm
import wandb

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse, indexes2oneHot
from lib.corruption import CorruptionMeta
from lib.spSet import SpeechCommandsV2C
from lib.dataset import IdxSet, PseudoLabelSet, Subset
from lib.component import Components, FrequenceTokenTransformer, AmplitudeToDB
from lib.adaptation import collect_worst_item
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import CrossEntropyLabelSmooth
from ..utils import build_model, load_weight

def pseudo_labeling(
    args:argparse.Namespace, teach_auts:list[nn.Module], teach_clsfs:list[nn.Module], std_aut:nn.Module,
    std_clsf:nn.Module, corruption_types:list[str], data_tfs:list[nn.Module]
):
    for teach_aut in teach_auts: teach_aut.eval()
    for teach_clsf in teach_clsfs: teach_clsf.eval()
    std_aut.eval(); std_clsf.eval()
    print('Pseudo labeling...')
    sc2c_set = SpeechCommandsV2C(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types, 
        data_tf=data_tfs
    )
    sc2c_set = IdxSet(sc2c_set)
    sc2c_loader = DataLoader(
        dataset=sc2c_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )
    teach_local_corrs, std_local_corrs = {it:0. for it in corruption_types}, {it:0. for it in corruption_types}
    teach_local_sizes, std_local_sizes = {it:0 for it in corruption_types}, {it:0 for it in corruption_types}
    final_pred, idxs = [], []
    output_cache = {}
    ttl_corr, ttl_size = 0., 0.
    for j, data in tqdm(enumerate(sc2c_loader), total=len(sc2c_loader)):
        labels = data[-1]
        for i in range(1, len(data)-1):
            features = data[i].to(args.device)
            corruption_type = corruption_types[i-1]
            with torch.inference_mode():
                outputs, _ = std_clsf(std_aut(features)[0])
                outputs = outputs.detach().cpu()
                _, preds = torch.max(outputs, dim=1)
            std_local_corrs[corruption_type] += (preds==labels).sum().item()
            std_local_sizes[corruption_type] += labels.shape[0]

            preds = indexes2oneHot(labels=preds, class_num=args.class_num)
            preds = preds * (1/len(corruption_type))
            if i==1: pred_cache = preds
            else: pred_cache = pred_cache + preds

            teach_aut, teach_clsf = teach_auts[i-1], teach_clsfs[i-1]
            with torch.inference_mode():
                outputs, _ = teach_clsf(teach_aut(features)[0])
                outputs = outputs.detach().cpu()
                _, preds = torch.max(outputs, dim=1)
            teach_local_corrs[corruption_type] += (preds==labels).sum().item()
            teach_local_sizes[corruption_type] += labels.shape[0]
            preds = indexes2oneHot(labels=preds, class_num=args.class_num)
            preds = preds * args.elect_weights[corruption_type]
            pred_cache = pred_cache + preds
            if j == 0: output_cache[corruption_type] = [outputs]
            else: output_cache[corruption_type].append(outputs)
        _, pred_cache = torch.max(pred_cache, dim=1)
        ttl_corr += (pred_cache==labels).sum().item()
        ttl_size += labels.shape[0]
        if j == 0:
            final_pred = [pred_cache]
            idxs = [data[0]]
        else: 
            final_pred.append(pred_cache)
            idxs.append(data[0])
    teach_local_accus, std_local_accus = {}, {}
    for corruption_type in corruption_types:
        teach_local_accus[corruption_type] = teach_local_corrs[corruption_type]/teach_local_sizes[corruption_type]
        std_local_accus[corruption_type] = std_local_corrs[corruption_type]/std_local_sizes[corruption_type]
    teach_global_accu = sum([v for k,v in teach_local_corrs.items()])/sum([v for k,v in teach_local_sizes.items()])
    std_global_accu = sum([v for k,v in std_local_corrs.items()])/sum([v for k,v in std_local_sizes.items()])
    pseudo_accu = ttl_corr/ttl_size
    print(f'Pseudo_labeling accuracy is:{pseudo_accu:.4f}')
    print(f'Student Adaptation global accuray is: {std_global_accu:.4f}')
    print('Student Adaptation local accuracies are', {k:round(v, ndigits=4) for k,v in std_local_accus.items()})
    print(f'Teacher Adaptation global accuracy is: {teach_global_accu:.4f}')
    print('Teacher Adaptation local accuracies are', {k:round(v, ndigits=4) for k,v in teach_local_accus.items()})
    final_pred = torch.concat(final_pred, dim=0)
    idxs = torch.concat(idxs, dim=0)
    tmp = {}
    for k, v in output_cache.items():
        tmp[k] = torch.concat(v, dim=0)
    output_cache = tmp
    return idxs, final_pred, output_cache, teach_global_accu, std_global_accu

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
    ap.add_argument('--num_of_shft', type=int, default=3, help='maximum number of shifting teachers')
    ap.add_argument('--fail_coll_lim', type=int, default=3, help='maximum number of fail prediction be choosed in worst list')

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--teach_aut_lr_decay', type=float, default=1.0)
    ap.add_argument('--teach_clsf_lr_decay', type=float, default=1.0)
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
    args.target_length=104

    print('Initialization...')
    teach_auts, teach_clsfs = [], []
    std_aut, std_clsf = build_model(args=args)
    load_weight(args=args, aut=std_aut, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION,)
    teach_optimizers = []
    teach_loss_fun = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available())
    for corruption_type in tqdm(corruption_types):
        teach_aut, teach_clsf = build_model(args=args)
        load_weight(
            args=args, aut=teach_aut, clsf=teach_clsf, mode='adaptation', 
            metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level)
        )
        teach_auts.append(teach_aut)
        teach_clsfs.append(teach_clsf)
        optimizer = build_optimizer(
            lr=args.lr, auT=teach_aut, auC=teach_clsf, auT_decay=args.teach_aut_lr_decay, 
            auC_decay=args.teach_clsf_lr_decay
        )
        teach_optimizers.append(optimizer)

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
        idxs, final_pred, output_cache, teach_global_accu, std_global_accu = pseudo_labeling(
            args=args, teach_auts=teach_auts, teach_clsfs=teach_clsfs, std_aut=std_aut, std_clsf=std_clsf, 
            corruption_types=corruption_types, data_tfs=data_tfs
        )
        # TODO: store the highest accuracy
        worst_list, shft_typs = collect_worst_item(
            args=args, corruption_types=corruption_types, idx_cache=idxs, pred_cache=final_pred, 
            output_cache=output_cache, step=epoch, logger=wandb_run
        )
        if epoch == args.max_epoch: break
        print('Adapting...')
        for teach_aut in teach_auts: teach_aut.train()
        for teach_clsf in teach_clsfs: teach_clsf.train()
        std_aut.train(); std_clsf.train()
        for idx, corruption_type in tqdm(enumerate(corruption_types), total=len(corruption_types)):
            # teacher adaptation
            sc2c_set = SpeechCommandsV2C(
                root_path=args.adpt_set_path, corruption_level=args.corruption_level, 
                corruption_type=corruption_type, data_tf=data_tfs[0]
            )
            sc2c_set = PseudoLabelSet(dataset=sc2c_set, pseudo_labels=worst_list[corruption_type], label_position=1)
            sc2c_set = Subset(dataset=sc2c_set, id_list=list(worst_list[corruption_type].keys()))
            sc2c_loader = DataLoader(
                dataset=sc2c_set, batch_size=args.batch_size, shuffle=True, drop_last=False, 
                num_workers=args.num_workers
            )
            teach_aut, teach_clsf = teach_auts[idx], teach_clsfs[idx]
            teach_optimizer = teach_optimizers[idx]

            for features, labels in sc2c_loader:
                if corruption_type not in shft_typs: break
                features, labels = features.to(args.device), labels.to(args.device)  
                if features.shape[0] == 1:
                    features = features.repeat(4, 1, 1)
                    labels = labels.repeat(4, 1) 

                outputs, _ = teach_clsf(teach_aut(features)[0])
                _, preds = torch.max(labels, dim=1)
                loss = teach_loss_fun(outputs, preds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % args.interval == 0:
                lr_scheduler(
                    optimizer=optimizer, epoch=epoch+1, lr_cardinality=args.lr_cardinality,
                    gamma=args.lr_gamma, threshold=args.lr_threshold, momentum=args.lr_momentum
                )

        exit()
    wandb.finish()
    print('END!')