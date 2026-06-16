import argparse
import json
import numpy as np
import random 
import os
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import make_unless_exits, print_argparse
from lib.loss import CrossEntropyLabelSmooth
from lib.corruption import CorruptionMeta
from lib.optimizer import build_optimizer, lr_scheduler
from lib.component import Components, AudioClip, ReduceChannel
from lib.enSet import UrbanSound8KC
from lib.dataset import IdxSet, PseudoLabelSet, Subset
from lib.adaptation import collect_worst_item
from ..utils import build_model, teach_inference
from PANNs.lib.utils import load_weight, store_weight, pan_freeze

def pseudo_labeling(
        args:argparse.Namespace, pans:list[nn.Module], clsfs:list[nn.Module], data_tfs:list[nn.Module],
        corruption_types:list[str], step:int, logger
    ):
    print("Pseudo-labeling...")
    for pan in pans: pan.eval()
    for clsf in clsfs: clsf.eval()
    output_cache = {}

    us8c = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs
    )
    us8c = IdxSet(dataset=us8c)
    us8_loader = DataLoader(
        dataset=us8c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    output_cache = {k:[] for k in corruption_types}
    y_t, y_p = [], []
    for j, data in tqdm(enumerate(us8_loader), total=len(us8_loader)):
        labels = data[-1]
        idxs = data[0]
        for i in range(1, len(data)-1):
            features = data[i].to(args.device)
            pan = pans[i-1]
            clsf = clsfs[i-1]
            corruption_type = corruption_types[i-1]
            with torch.inference_mode():
                outputs = clsf(pan(features)['embedding'])
                outputs = outputs.detach().cpu()
            preds = nn.functional.softmax(outputs, dim=1) * args.elect_weights[corruption_type]
            if i == 1: final_preds = preds
            else: final_preds += preds
            if j == 0: output_cache[corruption_type] = [outputs]
            else: output_cache[corruption_type].append(outputs)
        _, final_preds = torch.max(final_preds, dim=1)
        y_t.append(labels)
        y_p.append(final_preds)
        if j == 0: 
            pred_cache = [final_preds]
            idx_cache = [idxs]
        else:
            pred_cache.append(final_preds)
            idx_cache.append(idxs)
    y_t = torch.concat(y_t, dim=0)
    y_p = torch.concat(y_p, dim=0)
    pseudo_f1 = f1_score(y_true=y_t.numpy(), y_pred=y_p.numpy(), average='macro')
    print(f'Teacher election pseudo-labeling F1 score is: {pseudo_f1:.4f}')
    logger.log(data={'Adaptation/pseudo-labeling F1 score': pseudo_f1}, step=step)

    # Merging output cache
    tmp = {}
    for key, value in output_cache.items():
        tmp[key] = torch.concat(value, dim=0)
    output_cache = tmp
    pred_cache = torch.concat(pred_cache, dim=0)
    idx_cache = torch.concat(idx_cache, dim=0)
    return output_cache, pred_cache, idx_cache, pseudo_f1

def teacher_f1_analyzing(
        args:argparse.Namespace, pans:list[nn.Module], clsfs:list[nn.Module], corruption_types:list[str],
        data_tfs:list[nn.Module], step:int, logger
    ):
    print('Teacher accuracy analyzing...')
    print('Adaptation Set')
    adpt_us8c = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs
    )
    adpt_loader = DataLoader(
        dataset=adpt_us8c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    adpt_f1s = teach_inference(
        args=args, corruption_types=corruption_types, pans=teach_pans, clsfs=teach_clsfs, data_loader=adpt_loader
    )
    print({k:round(v, ndigits=4) for k,v in adpt_f1s.items()})
    logger.log(data={f'Adaptation/{k} F1 score': v for k, v in adpt_f1s.items()}, step=step)

    print('Evaluation set')
    eval_us8c = UrbanSound8KC(
        root_path=args.eval_set_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_us8c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    eval_f1s = teach_inference(
        args=args, corruption_types=corruption_types, pans=teach_pans, clsfs=teach_clsfs, data_loader=eval_loader
    )
    print({k:round(v, ndigits=4) for k,v in eval_f1s.items()})
    logger.log(data={f'Evaluation/{k} F1 score': v for k, v in eval_f1s.items()}, step=step)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
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
    ap.add_argument('--lr_momentums', type=str)
    ap.add_argument('--pan_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--interval', type=int, default=1, help='interval number')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = constants.pann_sample_rate
        args.orig_sample_rate = 44100
        args.audio_length = int(4 * constants.pann_sample_rate)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'PANNs'
    args.elect_weights = json.loads(args.elect_weights)
    args.lr_momentums = json.loads(args.lr_momentums)
    if not args.forbid_ls.strip():  args.forbid_ls = []
    else: args.forbid_ls = [k.strip() for k in args.forbid_ls.split(',')]
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
        config=args, tags=['Audio Classification', 'Teacher Adaptation', args.dataset]
    )

    print("Initialization...")
    corruption_types=['WHN', 'ENSC', 'PSH', 'TST']
    teach_pans, teach_clsfs = [], []
    max_pseudo_accu = 0.
    optimizers = []
    loss_fun = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available())
    for corruption_type in tqdm(corruption_types):
        cmeta = CorruptionMeta(type=corruption_type, level=args.corruption_level)
        teach_pan, teach_clsf = build_model(args=args, use_pre_weight=False)
        load_weight(args=args, panns=teach_pan, clsf=teach_clsf, mode='adaptation', metaInfo=cmeta)
        teach_pans.append(teach_pan); teach_clsfs.append(teach_clsf)
        optimizer = build_optimizer(
            lr=args.lr, auT=teach_pan, auC=teach_clsf, auT_decay=args.pan_lr_decay, 
            auC_decay=args.clsf_lr_decay
        )
        optimizers.append(optimizer)

    print("Preparing datasets...")
    data_tfs = [Components(transforms=[
        Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
        AudioClip(max_length=args.audio_length, mode='head', is_random=False),
        ReduceChannel()
    ])] * len(corruption_types)

    max_pl_f1 = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        teacher_f1_analyzing(
            args=args, pans=teach_pans, clsfs=teach_clsfs, corruption_types=corruption_types, 
            data_tfs=data_tfs, step=epoch, logger=wandb_run
        )
        output_cache, pred_cache, idx_cache, pseudo_f1 = pseudo_labeling(
            args=args, pans=teach_pans, clsfs=teach_clsfs, data_tfs=data_tfs, corruption_types=corruption_types,
            step=epoch, logger=wandb_run
        )
        if max_pl_f1 <= pseudo_f1:
            max_pl_f1 = pseudo_f1
            for i, corruption_type in enumerate(corruption_types):
                store_weight(
                    args=args, panns=teach_pans[i], clsf=teach_clsfs[i], mode='adaptation', root_path=args.output_path,
                    metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level)
                )
        worst_list, shft_typs = collect_worst_item(
            args=args, corruption_types=corruption_types, idx_cache=idx_cache, pred_cache=pred_cache, 
            output_cache=output_cache, step=epoch, logger=wandb_run, feature_label=False
        )
        if epoch == args.max_epoch: break
        print('Adapting...')
        for teach_pan in teach_pans: 
            teach_pan.train()
            pan_freeze(pan=teach_pan, batch1d=True, batch2d=True)
        for teach_clsf in teach_clsfs: teach_clsf.train()
        for idx, corruption_type in tqdm(enumerate(corruption_types), total=len(corruption_types), position=0):
            adpt_set = UrbanSound8KC(
                root_path=args.adpt_set_path, corruption_type=corruption_type, corruption_level=args.corruption_level,
                data_tf=data_tfs[0]
            )
            adpt_set = PseudoLabelSet(dataset=adpt_set, pseudo_labels=worst_list[corruption_type], label_position=1)
            adpt_set = Subset(dataset=adpt_set, id_list=list(worst_list[corruption_type].keys()))
            adpt_loader = DataLoader(
                dataset=adpt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
            )
            teach_pan, teach_clsf = teach_pans[idx], teach_clsfs[idx]
            optimizer = optimizers[idx]

            for features, labels in tqdm(adpt_loader, desc=f'{corruption_type}-{args.corruption_level}', position=1, leave=False):
                if corruption_type not in shft_typs: break
                if corruption_type in args.forbid_ls: break
                features, labels = features.to(args.device), labels.to(args.device)
                if features.shape[0] == 1:
                    features = features.repeat(4, 1)
                    labels = labels.repeat(4)

                outputs = teach_clsf(teach_pan(features)['embedding'])
                clsf_loss = loss_fun(outputs, labels)

                optimizer.zero_grad()
                clsf_loss.backward()
                optimizer.step()

            if epoch % args.interval == 0:
                lr_scheduler(
                    optimizer=optimizer, epoch=epoch+1, lr_cardinality=args.lr_cardinality,
                    gamma=args.lr_gamma, threshold=args.lr_threshold, momentum=args.lr_momentums[corruption_type]
                )
    wandb_run.finish()
    print('END!')