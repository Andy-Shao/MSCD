import argparse
import os
import numpy as np
import random
import wandb
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample

from lib import constants
from lib.utils import print_argparse, make_unless_exits, store_model_structure_to_txt
from lib.utils import ConfigDict
from lib.spSet import SpeechCommandsV2
from lib.component import Components, ReduceChannel, AudioPadding
from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.lib.utils import __cal_model_path__
from PANNs.classifier import PannClassifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--hub_lr_decay', type=float, default=1.0)
    ap.add_argument('--clsf_lr_decay', type=float, default=1.0)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--max_epoch', type=int, default=30)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default='2026')
    ap.add_argument('--smooth', type=float, default=.1)

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.arch = 'PANNs'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'train')

    torch.backends.cudnn.benchmark == True
    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

    print_argparse(args=args)
    ############################################################

    make_unless_exits(args.output_path)
    make_unless_exits(args.dataset_root_path)

    wandb_run = wandb.init(
        project=f'{constants.PROJECT_TITLE}-{constants.TRAIN_TAG}', 
        name=f'{constants.architecture_dic[args.arch]}-{constants.dataset_dic[args.dataset]}', mode='online' if args.wandb else 'disabled', 
        config=args, tags=['Audio Classification', 'Test-time Adaptation', args.dataset]
    )
    pan_sr = 32000
    pan = Wavegram_Logmel_Cnn14(
        sample_rate=pan_sr, 
        window_size=1024,
        hop_size=320,
        mel_bins=64, 
        fmin=50,
        fmax=14000,
        classes_num=527 #audio set
    )
    ckpt = hf_hub_download(
        repo_id='nicofarr/panns_Wavegram_Logmel_Cnn14',
        filename='model.safetensors'
    )
    state = load_file(ckpt)
    # Remove 'backbone.' prefix
    state = {k.replace("backbone.", ""): v for k, v in state.items()}
    pan.load_state_dict(state_dict=state)
    cfg = ConfigDict()
    cfg.class_num = args.class_num
    cfg.embed_num = 2048
    clsf = PannClassifier(config=cfg)

    pan, clsf = pan.to(device=args.device), clsf.to(device=args.device)

    pan_pth, clsf_pth = __cal_model_path__(args=args, mode='origin', root_path=args.output_path)
    pan_pth, clsf_pth = pan_pth.replace('.pt', '.txt'), clsf_pth.replace('.pt', '.txt')
    store_model_structure_to_txt(model=pan, output_path=pan_pth)
    store_model_structure_to_txt(model=clsf, output_path=clsf_pth)

    pan.eval(); clsf.eval()

    train_set = SpeechCommandsV2(
        root_path=args.dataset_root_path, mode='training', download=True, 
        data_tf=Components(transforms=[
            Resample(orig_freq=args.sample_rate, new_freq=pan_sr),
            AudioPadding(max_length=pan_sr, sample_rate=pan_sr, random_shift=False),
            ReduceChannel()
        ])
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, 
        num_workers=args.num_workers
    )
    for features, labels in tqdm(train_loader):
        features = features.to(args.device)
        with torch.inference_mode():
            outputs = clsf(pan(features)['embedding'])
    print('END!')