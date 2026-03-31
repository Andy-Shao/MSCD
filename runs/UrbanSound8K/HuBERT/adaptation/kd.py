import argparse
import json
import os
import numpy as np
import random
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import print_argparse, make_unless_exits
from lib.component import Components, AudioClip, ReduceChannel
from lib.corruption import CorruptionMeta
from lib.enSet import UrbanSound8KC
from lib.dataset import IdxSet, PseudoLabelSet
from lib.optimizer import build_optimizer, lr_scheduler
from lib.loss import ContrastiveLoss
from ..utils import build_model, mlt_inference
from HuBERT.lib.utils import load_weight, store_weight

def student_f1_analyzing(
    args:argparse.Namespace, hub:nn.Module, clsf:nn.Module, corruption_types:list[str],
    data_tfs:list[nn.Module], step:int, logger
) -> float:
    print('Adapataion set accuracy analyzing...')
    adpt_us8c = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    adpt_loader = DataLoader(
        dataset=adpt_us8c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    adpt_global_f1, adpt_local_f1 = mlt_inference(
        args=args, corruption_types=corruption_types, hub=hub, clsf=clsf, data_loader=adpt_loader,
    )
    print('Adaptation local F1 scores are:', {key: round(value, ndigits=4) for key, value in adpt_local_f1.items()})
    print(f'Adaptation global F1 score is: {adpt_global_f1:.4f}')
    logger.log(data={f'Adaptation/{k} F1 score': v for k,v in adpt_local_f1.items()}, step=step)
    logger.log(data={f'Adaptation/Global F1 score': adpt_global_f1}, step=step)

    print('Evaluation set accuracy analyzing...')
    eval_us8c = UrbanSound8KC(
        root_path=args.eval_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    eval_loader = DataLoader(
        dataset=eval_us8c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    eval_global_f1, eval_local_f1 = mlt_inference(
        args=args, corruption_types=corruption_types, hub=hub, clsf=clsf, data_loader=eval_loader
    )
    print('Evaluation local F1 scores are:', {key: round(value, ndigits=4) for key, value in eval_local_f1.items()})
    print(f'Evaluation global F1 score is: {eval_global_f1:.4f}')
    logger.log(data={f'Evaluation/{k} F1 score': v for k,v in eval_local_f1.items()}, step=step)
    logger.log(data={f'Evaluation/Global F1 score': eval_global_f1}, step=step)
    return adpt_global_f1

def pseudo_labeling(args:argparse.Namespace, corruption_types:list[str], data_tfs:list[nn.Module]):
    print('Loading all teachers...')
    teach_hubs, teach_clsfs = [], []
    for corruption_type in tqdm(corruption_types):
        teach_hub, teach_clsf = build_model(args=args)
        load_weight(
            args=args, hubert=teach_hub, clsf=teach_clsf, mode='adaptation',
            metaInfo=CorruptionMeta(type=corruption_type, level=args.corruption_level),
        )
        teach_hub.eval(); teach_clsf.eval()
        teach_hubs.append(teach_hub)
        teach_clsfs.append(teach_clsf)

    print('Pseudo-labeling...')
    us8_c = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    us8_c = IdxSet(dataset=us8_c)
    us8_loader = DataLoader(
        dataset=us8_c, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    idx_cache, pred_cache = [], []
    y_ts, y_ps = [], []
    for k, data in tqdm(enumerate(us8_loader), total=len(us8_loader)):
        idxs = data[0]
        labels = data[-1]
        for j in range(1, len(data)-1):
            features = data[j].to(args.device)
            teach_hub = teach_hubs[j-1]
            teach_clsf = teach_clsfs[j-1]
            corruption_type = corruption_types[j-1]
            with torch.inference_mode():
                outputs = teach_clsf(teach_hub(features)[0])
                outputs = outputs.detach().cpu()
            preds = nn.functional.softmax(outputs, dim=1) * args.elect_weights[corruption_type]
            if j == 1: final_preds = preds
            else: final_preds = final_preds + preds
        y_ts.append(labels)
        y_ps.append(torch.max(final_preds, dim=1)[1])
        idx_cache.append(idxs)
        pred_cache.append(final_preds)
    idx_cache = torch.concat(idx_cache, dim=0)
    pred_cache = torch.concat(pred_cache, dim=0)
    y_ts = torch.concat(y_ts, dim=0)
    y_ps = torch.concat(y_ps, dim=0)
    pl_f1 = f1_score(y_true=y_ts.numpy(), y_pred=y_ps.numpy(), average='macro')
    print(f'Teacher election pseudo-labeling F1 score is: {pl_f1:.4f}')
    teach_auts = None; teach_clsfs = None
        
    print('Calculating pseudo-labels...')
    pseudo_labels = {} # key -> idx, value -> smoothed label
    for i in tqdm(range(len(idx_cache)), total=len(idx_cache)):
        idx = int(idx_cache[i].item())
        pred = pred_cache[i]
        max_val, max_pos = torch.max(pred, dim=0)
        pred = torch.eye(args.class_num)[max_pos]
        if args.pseudo_threshold > max_val:
            pseudo_smooth = args.lw_def_smth
        else:
            pseudo_smooth = args.hi_def_smth
        pred = (1-pseudo_smooth)*pred + pseudo_smooth/args.class_num
        pseudo_labels[idx] = pred
    return pseudo_labels

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='UrbanSound8K', choices=['UrbanSound8K'])
    ap.add_argument('--adpt_set_path', type=str)
    ap.add_argument('--eval_set_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_pth', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--elect_weights', type=str)
    ap.add_argument('--max_epoch', type=int, default=20)
    ap.add_argument('--pseudo_threshold', type=float, default=6.0)
    ap.add_argument('--hi_def_smth', type=float, default=.1)
    ap.add_argument('--lw_def_smth', type=float, default=.2)
    ap.add_argument('--ctr_rt', type=float, default=1.)
    ap.add_argument('--ctr_dist', type=str, default='l2', choices=['cos_sim', 'l2', 'sq_l2'])

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--lr_momentum', type=float, default=.9)
    ap.add_argument('--interval', type=int, default=1, help='interval number')

    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2026, help='random seed')

    args = ap.parse_args()
    if args.dataset == 'UrbanSound8K':
        args.class_num = 10
        args.sample_rate = 16000
        args.orig_sample_rate = 44100
        args.audio_length = int(4 * 16000)
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
    args.elect_weights = json.loads(args.elect_weights)
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, constants.STUDENT_ADAPTATION)
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################
    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.STUDENT_ADAPTATION}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}-{args.corruption_level}', 
        mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Knowledge Distillation', args.dataset]
    )

    print("Initialization...")
    corruption_types=['WHN', 'ENSC', 'PSH', 'TST']
    data_tfs = [Components(transforms=[
        Resample(orig_freq=args.orig_sample_rate, new_freq=args.sample_rate),
        AudioClip(max_length=args.audio_length, mode='head', is_random=False),
        ReduceChannel(),
    ])] * len(corruption_types)
    pseudo_labels = pseudo_labeling(
        args=args, corruption_types=corruption_types, data_tfs=data_tfs
    )
    adpt_set = UrbanSound8KC(
        root_path=args.adpt_set_path, corruption_level=args.corruption_level, corruption_type=corruption_types,
        data_tf=data_tfs
    )
    adpt_set = PseudoLabelSet(dataset=adpt_set, pseudo_labels=pseudo_labels, label_position=len(corruption_types))
    adpt_loader = DataLoader(
        dataset=adpt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )
    std_hub, std_clsf = build_model(args=args)
    load_weight(args=args, hubert=std_hub, clsf=std_clsf, mode='origin')
    optimizer = build_optimizer(
        lr=args.lr, auT=std_hub, auC=std_clsf, auT_decay=args.hub_lr_decay, auC_decay=args.clsf_lr_decay
    )
    ctr_loss_fun = ContrastiveLoss(
        hi_def_smth=args.hi_def_smth, class_num=args.class_num, device=args.device, dist=args.ctr_dist
    )

    print('Student Adaptation')
    max_f1 = 0.
    for epoch in range(args.max_epoch+1):
        print(f'Epoch: {epoch+1}/{args.max_epoch} processing...')
        print('Inferencing...')
        f1 = student_f1_analyzing(
            args=args, hub=std_hub, clsf=std_clsf, corruption_types=corruption_types, data_tfs=data_tfs, 
            step=epoch, logger=wandb_run
        )
        if max_f1 <= f1:
            max_f1 = f1
            store_weight(
                args=args, hubert=std_hub, clsf=std_clsf, mode=constants.STUDENT_ADAPTATION, root_path=args.output_path
            )
        
        if epoch >= args.max_epoch: break
        print('Adaptating...')
        std_hub.train(); std_clsf.train()
        # hub_clsf_freeze(model=std_clsf)
        ttl_loss = 0.; ttl_clsf_loss = 0.; ttl_ctr_loss = 0.
        for adpt_data in tqdm(adpt_loader):
            labels = adpt_data[-1].to(args.device)
            optimizer.zero_grad()
            for i in range(len(adpt_data)-1):
                features = adpt_data[i].to(args.device)

                outputs = std_clsf(std_hub(features)[0])

                # classification loss
                clsf_loss = (-labels * nn.functional.log_softmax(outputs, dim=1)).sum(dim=1) # cross-entropy loss
                clsf_loss = clsf_loss.mean()

                # contrastive loss
                if args.ctr_rt > 0.:
                    ctr_loss = args.ctr_rt * ctr_loss_fun(outputs, labels)
                else: ctr_loss = torch.tensor(0.).to(device=args.device)

                loss = (clsf_loss + ctr_loss) / (len(adpt_data)-1)
                loss.backward()
                ttl_loss += loss.detach().cpu().item()
                ttl_clsf_loss += clsf_loss.detach().cpu().item()
                ttl_ctr_loss += ctr_loss.detach().cpu().item()
            optimizer.step()
        wandb_run.log(data={
            'Loss/TTL Loss': ttl_loss/(len(adpt_loader)),
            'Loss/Classification loss': ttl_clsf_loss/(len(adpt_loader)*len(corruption_types)),
            'Loss/Contrastive loss': ttl_ctr_loss/(len(adpt_loader)*len(corruption_types)),
        }, step=epoch)
        if epoch % args.interval == 0:
            lr_scheduler(
                optimizer=optimizer, epoch=epoch+1, lr_cardinality=args.lr_cardinality,
                gamma=args.lr_gamma, threshold=args.lr_threshold, momentum=args.lr_momentum
            )
    wandb_run.finish()
    print('END!')