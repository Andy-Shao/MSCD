import argparse
import os
import wandb
import numpy as np
import random
from tqdm import tqdm
import json

from lib import constants
from lib.utils import make_unless_exits, print_argparse, indexes2oneHot

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

from lib.dataset import IdxSet, Subset, PseudoLabelSet
from lib.spSet import SpeechCommandsV2C
from lib.corruption import CorruptionMeta
from lib.component import Components, AmplitudeToDB, FrequenceTokenTransformer
from lib.optimizer import build_optimizer, lr_scheduler
from ..utils import build_model, load_weight, inference

def teacher_accu_analyzing(
        args:argparse.Namespace, auts:list[nn.Module], clsfs:list[nn.Module], corruption_types:list[str],
        data_tf:nn.Module, step:int, logger
    ):
    print('Teacher accuracy analyzing...')
    for aut in auts: aut.eval()
    for clsf in clsfs: clsf.eval()
    accu_dic = {}
    for idx, corruption_type in tqdm(enumerate(corruption_types), total=len(corruption_types)):
        sc2_c = SpeechCommandsV2C(
            root_path=args.dataset_root_path, corruption_level=args.corruption_level, 
            corruption_type=corruption_type, data_tf=data_tf
        )
        sc2_c_loader = DataLoader(
            dataset=sc2_c, batch_size=args.batch_size, shuffle=False, drop_last=False, 
            num_workers=args.num_workers
        )
        accu = inference(args=args, aut=auts[idx], clsf=clsfs[idx], data_loader=sc2_c_loader, tqdmable=False)
        logger.log(data={f'Accuracy/{corruption_type}-{args.corruption_level}': accu}, step=step)
        accu_dic[f'{corruption_type}-{args.corruption_level}']=round(accu, ndigits=4)
    print(accu_dic)

def collect_worst_item(
        args:argparse.Namespace, corruption_types:list[str], idx_cache:dict, pred_cache:dict,
        output_cache:dict, step:int, logger
    ) -> dict:
    """return dict: key -> corruption type, value -> [idxs, labels]"""
    print('Scanning and finding the most worst K teachers...')
    K = args.num_of_shft
    assert K < len(corruption_types)
    worst_list = {} # key -> corruption type, value -> [idxs, labels]
    for corruption_type in corruption_types:
        worst_list[corruption_type] = {}
    for idx, label, targets in tqdm(WorstItemSearch(
        idxs=idx_cache, preds=pred_cache, outputs=output_cache, K=K, corruption_types=corruption_types
    )):
        for target in targets:
            worst_item = worst_list[target]
            worst_item[idx] = label

    # print('Worst list presentation:')
    for corruption_type in corruption_types:
        print(f'type: {corruption_type}, size: {len(worst_list[corruption_type].keys())}')
        logger.log(data={f'WorstList/{corruption_type}':len(worst_list[corruption_type].keys())}, step=step)
    return worst_list

class WorstItemSearch:
    def __init__(
        self, idxs:torch.Tensor, preds:torch.Tensor, outputs:torch.Tensor, K:int,
        corruption_types:list[str]
    ):
        self.idxs = idxs
        self.preds = preds
        self.outputs = outputs
        assert K > 0, Exception('Unsupport')
        self.K = K
        self.corruption_types = corruption_types
        self.i = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i >= self.idxs.shape[0]:
            raise StopIteration
        pred = self.preds[self.i].item()
        tmp = []
        for corruption_type in self.corruption_types:
            opt = self.outputs[corruption_type][self.i, :]
            opt = torch.unsqueeze_copy(opt, 0)
            tmp.append(opt)
        tmp = torch.concatenate(tmp, dim=0)
        _, indices = torch.sort(tmp[:, pred].clone(), descending=False)
        label = tmp[indices[-1], :]

        fail_check = (torch.max(tmp, dim=1)[1] != torch.tensor(([pred]*tmp.shape[0])))
        num_fail = fail_check.sum().item()
        idx = self.idxs[self.i].item()
        self.i += 1
        if num_fail == 0:
            return self.__next__()
        elif num_fail <= self.K:
            fail_idx = torch.where(fail_check)[0]
            targets = [corruption_types[k] for k in fail_idx]
        else:
            tmp[~fail_check] = 0.
            _, indices = torch.sort(tmp[:, pred].clone(), descending=False)
            targets = [corruption_types[k] for k in indices[len(indices)-num_fail:len(indices)-(num_fail-self.K)]]
        return idx, label, targets

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
            _, preds = torch.max(outputs, dim=1)
            preds = indexes2oneHot(labels=preds, class_num=args.class_num)
            preds = preds * args.elect_weights[corruption_type]
            if i == 1: final_preds = preds
            else: final_preds = final_preds + preds
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
    print(f'Teacher election pseudo-labeling accuracy is: {ttl_corr/ttl_size:.4f}')
    logger.log(data={'Accuracy/pseudo-labeling': ttl_corr/ttl_size}, step=step)

    # Merging output cache
    tmp = {}
    for key, value in output_cache.items():
        tmp[key] = torch.concat(value, dim=0)
    output_cache = tmp
    pred_cache = torch.concat(pred_cache, dim=0)
    idx_cache = torch.concat(idx_cache, dim=0)
    return output_cache, pred_cache, idx_cache

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)
    ap.add_argument('--corruption_level', type=str, choices=['L1', 'L2'])
    ap.add_argument('--elect_weights', type=str)
    ap.add_argument('--num_of_shft', type=int, default=3)
    ap.add_argument('--max_epoch', type=int, default=20)

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
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
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
    max_accus = {}
    optimizers = []
    for corruption_type in tqdm(corruption_types):
        cmeta = CorruptionMeta(type=corruption_type, level=args.corruption_level)
        aut, clsf = build_model(args=args)
        load_weight(args=args, aut=aut, clsf=clsf, mode='adaptation', metaInfo=cmeta)
        auts.append(aut)
        clsfs.append(clsf)
        max_accus[corruption_type] = 0.
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
        root_path=args.dataset_root_path, corruption_type=corruption_types, corruption_level=args.corruption_level,
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
        output_cache, pred_cache, idx_cache = pseudo_labeling(
            args=args, auts=auts, clsfs=clsfs, data_loader=sc2_c_loader, corruption_types=corruption_types,
            step=epoch, logger=wandb_run
        )

        worst_list = collect_worst_item(
            args=args, corruption_types=corruption_types, idx_cache=idx_cache, 
            pred_cache=pred_cache, output_cache=output_cache, step=epoch,
            logger=wandb_run
        )
        if epoch == args.max_epoch: break
        print('Adapting...')
        for aut in auts: aut.train()
        for clsf in clsfs: clsf.train()
        for idx, corruption_type in tqdm(enumerate(corruption_types), total=len(corruption_types)):
            adpt_set = SpeechCommandsV2C(
                root_path=args.dataset_root_path, corruption_level=args.corruption_level, 
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
            adpt_set = Subset(dataset=adpt_set, label_list=list(worst_list[corruption_type].keys()))

            adpt_loader = DataLoader(
                dataset=adpt_set, batch_size=args.batch_size, shuffle=True, drop_last=False, 
                num_workers=args.num_workers
            )
            aut, clsf = auts[idx], clsfs[idx]
            optimizer = optimizers[idx]

            for features, labels in adpt_loader:
                features, labels = features.to(args.device), labels.to(args.device)    

                outputs, _ = clsf(aut(features)[0])
                # logsoft_nll
                outputs = nn.functional.log_softmax(outputs, dim=1)
                _, preds = torch.max(labels, dim=1)
                clsf_loss = nn.NLLLoss(reduce='mean')(outputs, preds)

                optimizer.zero_grad()
                clsf_loss.backward()
                optimizer.step()

            learning_rate = optimizer.param_groups[0]['lr']
            if epoch % args.interval == 0:
                lr_scheduler(
                    optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality,
                    gamma=args.lr_gamma, threshold=args.lr_threshold, momentum=args.lr_momentum
                )

    wandb_run.finish()
    print('END!')