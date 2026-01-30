import argparse
import os
import pyrnnoise
import numpy as np

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
        root_path=args.dataset_root_path, corruption_level='L2', corruption_type='END1',
    )
    noisy, label = sc2_c[0]
    print(f'noisy wavform shape: {noisy.shape}, sample rate: {args.sample_rate}')

    torchaudio.save(
        uri=os.path.join('/root', 'output', 'noisy.wav'),
        src=noisy.detach(), 
        sample_rate=args.sample_rate,
        encoding='PCM_S',
        bits_per_sample=16
    )

    denoiser = pyrnnoise.RNNoise(sample_rate=args.sample_rate)
    noise_np = noisy * 32768.0
    noise_np = noise_np.clip(-32768.0, 32767.0)
    noise_np = noise_np.numpy()
    noise_np = noise_np.astype(np.int16)
    out = [denoise_audio for speech_prob, denoise_audio in denoiser.denoise_chunk(noise_np, partial=True)]
    clean_wav = torch.tensor(np.concat(out, axis=1)).to(torch.float32)/32768.0

    print(f'clean wavform shape:{clean_wav.shape}')

    torchaudio.save(
        uri=os.path.join('/root', 'output', 'clean.wav'),
        src=clean_wav.detach(),
        sample_rate=args.sample_rate,
        encoding='PCM_S',
        bits_per_sample=16
    )