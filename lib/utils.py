import os
import argparse

import torch
from torch import nn

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

def indexes2oneHot(labels:torch.Tensor, class_num:int) -> torch.Tensor:
    # ret = []
    # for idx in range(labels.shape[0]):
    #     ret.append(index2oneHot(label=labels[idx].item(), class_num=class_num).unsqueeze(dim=0))
    # return torch.cat(ret, dim=0)
    return torch.zeros(labels.shape[0], class_num).scatter_(1, labels.unsqueeze(1).cpu(), 1)

def index2oneHot(label:int, class_num:int) -> torch.Tensor:
    eyes = torch.eye(class_num)
    return eyes[label]

def hex_to_one_hot(hex_str:str, length:int) -> torch.Tensor:
    # Step 1: Hex to decimal
    decimal_val = int(hex_str, 16)
    
    # Step 2: Decimal to binary string
    binary_str = bin(decimal_val)[2:]
    
    # Step 3: Pad with zeros on the left
    binary_str = binary_str.zfill(length)
    
    # Step 4: Convert to list of ints
    one_hot = [float(bit) for bit in binary_str]
    
    one_hot = torch.tensor(one_hot)
    return one_hot

def one_hot_to_hex(one_hot:torch.Tensor) -> str:
    one_hot = one_hot.to(dtype=torch.int64).detach().numpy()
    # Convert to binary string
    binary_str = ''.join(str(bit) for bit in one_hot)
    
    # Convert binary string to decimal
    decimal_val = int(binary_str, 2)
    
    # Convert decimal to hexadecimal
    hex_val = hex(decimal_val)
    
    return hex_val

def make_unless_exits(url:str) -> None:
    if not os.path.exists(url):
        os.makedirs(url)

def print_argparse(args: argparse.Namespace) -> None:
    for arg in vars(args):
        print(f'--{arg} = {getattr(args, arg)}')

def count_ttl_params(model: nn.Module, filter_by_grad=False, requires_grad=True):
    if not filter_by_grad:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)
    
class ConfigDict:
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError as e:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'") from e

    def __setattr__(self, key, value):
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __repr__(self):
        return repr(self._data)