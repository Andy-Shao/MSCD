import argparse
from tqdm import tqdm

import torch

from lib.utils import unique_pairs

def is_hi_mark(out:torch.Tensor, pseudo_label: torch.Tensor, hi_def_smth:float, class_num:int) -> bool:
    hi_mark = (1-hi_def_smth)*torch.eye(class_num)[0] + hi_def_smth/class_num
    hi_mark, _ = torch.max(hi_mark, dim=0)
    pl_v, pl_pos = torch.max(pseudo_label, dim=0)
    out_v, out_pos = torch.max(out, dim=0)
    return pl_v >= hi_mark.item() and pl_pos == out_pos

def sim_mark(
    outs:torch.Tensor, pseudo_labels:torch.Tensor, hi_def_smth:float, 
    class_num: int, device:str
) -> torch.tensor:
    ret = []
    idx = torch.triu_indices(outs.shape[0], outs.shape[0], offset=1)
    idx = list(zip(idx[0].numpy(), idx[1].numpy()))
    for i,j in idx:
        i_check = is_hi_mark(
            out=outs[i], pseudo_label=pseudo_labels[i], hi_def_smth=hi_def_smth, class_num=class_num
        )
        j_check = is_hi_mark(
            out=outs[j], pseudo_label=pseudo_labels[j], hi_def_smth=hi_def_smth, class_num=class_num
        )
        if i_check and j_check:
            oi_v, oi_p = torch.max(outs[i], dim=0)
            oj_v, oj_p = torch.max(outs[j], dim=0)
            if oi_p == oj_p: ret.append(1.)
            else: ret.append(-1.)
        else: ret.append(0.)
    return torch.tensor(ret).to(device=device)

class WorstItemSearch:
    def __init__(
        self, idxs:torch.Tensor, preds:torch.Tensor, outputs:dict[str, torch.Tensor], K:int,
        corruption_types:list[str]
    ):
        self.idxs = idxs
        self.preds = preds
        self.outputs = outputs
        assert K > 0, 'Unsupport'
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
            targets = [self.corruption_types[k] for k in fail_idx]
        else:
            tmp[~fail_check] = 0.
            _, indices = torch.sort(tmp[:, pred].clone(), descending=False)
            targets = [self.corruption_types[k] for k in indices[len(indices)-num_fail:len(indices)-(num_fail-self.K)]]
        return idx, label, targets
    
def collect_worst_item(
        args:argparse.Namespace, corruption_types:list[str], idx_cache:dict, pred_cache:dict,
        output_cache:dict, step:int, logger
    ) -> tuple[dict[str, dict], list[str]]:
    """return dict: key -> corruption type, value -> [idxs, labels]"""
    print('Scanning and finding the most worst K teachers...')
    K = args.fail_coll_lim
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
    Q = args.num_of_shft
    assert Q < len(corruption_types), 'Unsupport!'
    shft_prio = {}
    for corruption_type in corruption_types:
        shft_prio[corruption_type] = len(worst_list[corruption_type].keys()) / args.elect_weights[corruption_type]
        print(f'type: {corruption_type}, size: {len(worst_list[corruption_type].keys())}, priority: {shft_prio[corruption_type]:.2f}')
        logger.log(data={f'WorstList/{corruption_type}':len(worst_list[corruption_type].keys())}, step=step)
    shft_typs = [it[0] for it in sorted(shft_prio.items(), key=lambda x: x[1], reverse=True)]
    shft_typs = shft_typs[0:Q]
    print(f'Shifting corruption types are: {shft_typs}')
    return worst_list, shft_typs