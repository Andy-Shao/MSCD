import os
import argparse

import torch
from torch import nn

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