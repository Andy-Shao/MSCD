import argparse
import os

import torch
import torchaudio

from lib.utils import print_argparse
from lib.spSet import SpeechCommandsV2C

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--batch_size', type=int, default=32)

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    
    print_argparse(args=args)
    ############################################################

    sc2_c = SpeechCommandsV2C(
        root_path=args.dataset_root_path, corruption_level='L2', corruption_type='WHN',
    )
    wavform, sample_rate = sc2_c[0]
    print(f'wavform shape: {wavform.shape}, sample rate: {sample_rate}')

    torchaudio.save(
        uri=os.path.join('/root', 'output', 'noise_case.wav'),
        src=wavform.detach(), 
        sample_rate=sample_rate,
        encoding='PCM_S',
        bits_per_sample=16
    )