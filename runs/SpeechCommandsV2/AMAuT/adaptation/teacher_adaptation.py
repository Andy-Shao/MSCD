import argparse
import os
import wandb
import numpy as np
import random
from tqdm import tqdm
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.adaptation import collect_worst_item, is_frozen
from lib.dataset import IdxSet, Subset, PseudoLabelSet
from lib.spSet import SpeechCommandsV2C
from lib.corruption import CorruptionMeta
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import CrossEntropyLabelSmooth
from ..utils import build_model, load_weight, inference, store_weight

def teacher_accu_analyzing(
        args:argparse.Namespace, auts:list[nn.Module], clsfs:list[nn.Module], corruption_types:list[str],
        data_tf:nn.Module, step:int, logger
    ):
    print('Teacher accuracy analyzing...')
    for aut in auts: aut.eval()
    for clsf in clsfs: clsf.eval()
    accu_dic = {}
    for idx, corruption_type in tqdm(enumerate(corruption_types), total=len(corruption_types)):
        adpt_set = SpeechCommandsV2C(
            root_path=args.adpt_set_path, corruption_level=args.corruption_level, 
            corruption_type=corruption_type, data_tf=data_tf
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
            num_workers=args.num_workers
        )
        accu = inference(args=args, aut=auts[idx], clsf=clsfs[idx], data_loader=adpt_loader, tqdmable=False)
        logger.log(data={f'Adaptation/{corruption_type} Accuracy': accu}, step=step)
        accu_dic[f'{corruption_type}-{args.corruption_level}']=round(accu, ndigits=4)

        eval_set = SpeechCommandsV2C(
            root_path=args.eval_set_path, corruption_level=args.corruption_level, 
            corruption_type=corruption_type, data_tf=data_tf
        )
        eval_loader = DataLoader(
            dataset=eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers
        )
        accu = inference(args=args, aut=auts[idx], clsf=clsfs[idx], data_loader=eval_loader, tqdmable=False)
        logger.log(data={f'Evaluation/{corruption_type} Accuracy': accu}, step=step)
    print(accu_dic)

def pseudo_labeling(
        args:argparse.Namespace, auts:list[nn.Module], clsfs:list[nn.Module], data_loader:DataLoader,
        corruption_types:list[str], step:int, logger
    ):
    print("Pseudo-labeling...")
    for aut in auts: aut.eval()
    for clsf in clsfs: clsf.eval()
    ttl_corr, ttl_size = 0., 0.
    output_cache = {}
    for j, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        labels = data[-1]
        idxs = data[0]
        for i in range(1, len(data)-1):
            features = data[i].to(args.device)
            aut = auts[i-1]
            clsf = clsfs[i-1]
            corruption_type = corruption_types[i-1]
            with torch.inference_mode():
                outputs, _ = clsf(aut(features)[0])
                outputs = outputs.detach().cpu()
            preds = nn.functional.softmax(outputs, dim=1) * args.elect_weights[corruption_type]
            if i == 1: final_preds = preds
            else: final_preds += preds
            if j == 0: output_cache[corruption_type] = [outputs]
            else: output_cache[corruption_type].append(outputs)
        _, final_preds = torch.max(final_preds, dim=1)
        ttl_corr += (final_preds==labels).sum().item()
        ttl_size += labels.shape[0]
        if j == 0: 
            pred_cache = [final_preds]
            idx_cache = [idxs]
        else: 
            pred_cache.append(final_preds)
            idx_cache.append(idxs)
    pseudo_accu = ttl_corr/ttl_size
    print(f'Teacher election pseudo-labeling accuracy is: {pseudo_accu:.4f}')
    logger.log(data={'Adaptation/pseudo-labeling Accuracy': pseudo_accu}, step=step)

    # Merging output cache
    tmp = {}
    for key, value in output_cache.items():
        tmp[key] = torch.concat(value, dim=0)
    output_cache = tmp
    pred_cache = torch.concat(pred_cache, dim=0)
    idx_cache = torch.concat(idx_cache, dim=0)
    return output_cache, pred_cache, idx_cache, pseudo_accu

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
    ap.add_argument('--rewgt_int', type=int, default=10)
    ap.add_argument('--rewgt_threshold', type=float, default=1.1)
    ap.add_argument('--num_of_shft', type=int, default=3, help='maximum number of shifting teachers')
    ap.add_argument('--fail_coll_lim', type=int, default=3, help='maximum number of fail prediction be choosed in worst list')
    ap.add_argument('--max_epoch', type=int, default=20)
    ap.add_argument('--forbid_ls', type=str, default="")
    ap.add_argument('--unfrz_pos', type=int, default=-1)
    ap.add_argument('--str_pos', type=int, default=10)

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
    args.target_length=104

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

    print("Preparing datasets...")
    data_tfs = [Components(transforms=[
        MelSpectrogram(
            sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            n_mels=args.n_mels, mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ])] * len(corruption_types)
    sc2_c = SpeechCommandsV2C(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs
    )
    sc2_c = IdxSet(dataset=sc2_c)
    sc2_c_loader = DataLoader(
        dataset=sc2_c, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers
    )

    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        teacher_accu_analyzing(
            args=args, auts=auts, clsfs=clsfs, corruption_types=corruption_types, step=epoch, logger=wandb_run,
            data_tf=Components(transforms=[
                MelSpectrogram(
                    sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                    n_mels=args.n_mels, mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                FrequenceTokenTransformer()
            ])
        )
        output_cache, pred_cache, idx_cache, pseudo_accu = pseudo_labeling(
            args=args, auts=auts, clsfs=clsfs, data_loader=sc2_c_loader, corruption_types=corruption_types,
            step=epoch, logger=wandb_run
        )
        if max_pseudo_accu <= pseudo_accu and epoch >= args.str_pos:
            max_pseudo_accu = pseudo_accu
            for i, corruption_type in enumerate(corruption_types):
                aut, clsf = auts[i], clsfs[i]
                store_weight(
                    args=args, aut=aut, clsf=clsf, mode='adaptation', 
                    metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level),
                    root_path=args.output_path
                )

        worst_list, shft_typs = collect_worst_item(
            args=args, corruption_types=corruption_types, idx_cache=idx_cache, 
            pred_cache=pred_cache, output_cache=output_cache, step=epoch,
            logger=wandb_run, feature_label=False
        )
        if epoch == args.max_epoch: break
        print('Adapting...')
        for aut in auts: aut.train()
        for clsf in clsfs: clsf.train()
        for idx, corruption_type in tqdm(enumerate(corruption_types), total=len(corruption_types), position=0):
            adpt_set = SpeechCommandsV2C(
                root_path=args.adpt_set_path, corruption_level=args.corruption_level, 
                corruption_type=corruption_type, data_tf=Components(transforms=[
                    MelSpectrogram(
                        sample_rate=args.sample_rate, n_fft=n_fft, win_length=win_length, 
                        hop_length=hop_length, n_mels=args.n_mels, mel_scale=mel_scale
                    ),
                    AmplitudeToDB(top_db=80., max_out=2.),
                    FrequenceTokenTransformer()
                ])
            )
            adpt_set = PseudoLabelSet(dataset=adpt_set, label_position=1, pseudo_labels=worst_list[corruption_type])
            adpt_set = Subset(dataset=adpt_set, id_list=list(worst_list[corruption_type].keys()))

            adpt_loader = DataLoader(
                dataset=adpt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, 
                num_workers=args.num_workers
            )
            aut, clsf = auts[idx], clsfs[idx]
            optimizer = optimizers[idx]

            for features, labels in tqdm(adpt_loader, desc=f'{corruption_type}-{args.corruption_level}', position=1, leave=False):
                if corruption_type not in shft_typs: break
                # if corruption_type in args.forbid_ls: break
                if is_frozen(args=args, epoch_num=epoch, crpt_typ=corruption_type): break
                features, labels = features.to(args.device), labels.to(args.device)  
                if features.shape[0] == 1:
                    features = features.repeat(4, 1, 1)
                    labels = labels.repeat(4)  

                outputs, _ = clsf(aut(features)[0])
                clsf_loss = loss_fun(outputs, labels)

                optimizer.zero_grad()
                clsf_loss.backward()
                optimizer.step()

            # learning_rate = optimizer.param_groups[0]['lr']
            if epoch % args.interval == 0:
                lr_scheduler(
                    optimizer=optimizer, epoch=epoch+1, lr_cardinality=args.lr_cardinality,
                    gamma=args.lr_gamma, threshold=args.lr_threshold, momentum=args.lr_momentum
                )
            if epoch % args.rewgt_int == 0 and epoch != 0:
                elect_weights = {}
                for k, wgt in args.elect_weights.items():
                    if wgt > 1.: 
                        elect_weight = ((wgt-1.)/2.) + 1.
                        if elect_weight < args.rewgt_threshold: elect_weight = 1.
                        elect_weights[k] = elect_weight
                    else: elect_weights[k] = wgt
                args.elect_weights = elect_weights
                args.rewgt_int = 5

    wandb_run.finish()
    print('END!')