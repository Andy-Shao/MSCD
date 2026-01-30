import os
import argparse

def make_unless_exits(url:str) -> None:
    if not os.path.exists(url):
        os.makedirs(url)

def print_argparse(args: argparse.Namespace) -> None:
    for arg in vars(args):
        print(f'--{arg} = {getattr(args, arg)}')