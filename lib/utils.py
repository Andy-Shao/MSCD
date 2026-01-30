import os
import argparse

from torch import nn

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