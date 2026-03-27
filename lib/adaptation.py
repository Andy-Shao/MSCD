import argparse
from tqdm import tqdm

import torch
from torch import nn

from HuBERT.lib.model import HuBClassifier

def hub_clsf_freeze(model:HuBClassifier):
    for component in model.modules():
        if isinstance(component, nn.BatchNorm1d):
            component.eval()

def amaut_freeze(model:nn.Module, batch:bool=True, drop:bool=True):
    for component in model.modules():
        if isinstance(component, nn.BatchNorm1d) and batch:
            component.eval()
        elif isinstance(component, nn.Dropout) and drop:
            component.eval()

def is_frozen(args:argparse.Namespace, epoch_num:int, crpt_typ:str) -> bool:
    if crpt_typ in args.forbid_ls:
        if args.unfrz_pos == -1: return True
        elif epoch_num >= args.unfrz_pos: return False
        else: return True
    else: return False

class WorstItemSearch:
    def __init__(
        self, idxs:torch.Tensor, preds:torch.Tensor, outputs:dict[str, torch.Tensor], K:int,
        corruption_types:list[str], feature_label:bool=True
    ):
        self.idxs = idxs
        self.preds = preds
        self.outputs = outputs
        assert K > 0, 'Unsupport'
        self.K = K
        self.corruption_types = corruption_types
        self.feature_label = feature_label
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
        if self.feature_label: label = tmp[indices[-1], :]
        else: label = int(pred)

        fail_check = (torch.max(tmp, dim=1)[1] != torch.tensor(([pred]*tmp.shape[0])))
        num_fail = fail_check.sum().item()
        idx = self.idxs[self.i].item()
        self.i += 1
        if num_fail == 0:
            return self.__next__()
        elif num_fail <= self.K:
            targets = [self.corruption_types[k] for k in indices[0:num_fail]]
        else:
            targets = [self.corruption_types[k] for k in indices[0:self.K]]
        return idx, label, targets
    
def collect_worst_item(
        args:argparse.Namespace, corruption_types:list[str], idx_cache:dict, pred_cache:dict,
        output_cache:dict, step:int, logger, feature_label:bool=True
    ) -> tuple[dict[str, dict], list[str]]:
    """return dict: key -> corruption type, value -> [idxs, labels]"""
    print('Scanning and finding the most worst K teachers...')
    K = args.fail_coll_lim
    assert K < len(corruption_types)
    worst_list = {} # key -> corruption type, value -> [idxs, labels]
    for corruption_type in corruption_types:
        worst_list[corruption_type] = {}
    for idx, label, targets in tqdm(WorstItemSearch(
        idxs=idx_cache, preds=pred_cache, outputs=output_cache, K=K, corruption_types=corruption_types,
        feature_label=feature_label
    )):
        for target in targets:
            worst_item = worst_list[target]
            worst_item[idx] = label

    # print('Worst list presentation:')
    Q = args.num_of_shft
    assert Q <= len(corruption_types), 'Unsupport!'
    shft_prio = {}
    for corruption_type in corruption_types:
        shft_prio[corruption_type] = len(worst_list[corruption_type].keys()) / args.elect_weights[corruption_type]
        print(f'type: {corruption_type}, size: {len(worst_list[corruption_type].keys())}, priority: {shft_prio[corruption_type]:.2f}')
        logger.log(data={f'WorstList/{corruption_type}':len(worst_list[corruption_type].keys())}, step=step)
    shft_typs = [it[0] for it in sorted(shft_prio.items(), key=lambda x: x[1], reverse=True)]
    shft_typs = shft_typs[0:Q]
    print(f'Shifting corruption types are: {shft_typs}')
    return worst_list, shft_typs