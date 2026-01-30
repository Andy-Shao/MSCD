import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.models import Wav2Vec2Model

from lib import constants
from lib.utils import make_unless_exits, print_argparse, count_ttl_params, ConfigDict
from lib.corruption import corruption_meta, CorruptionMeta
from lib.component import ReduceChannel, Components, RNNoiseTransform
from lib.dataset import TransferDataset
from lib.spSet import SpeechCommandsV2C
from AuT.lib.model import AudioClassifier

def denormalize(noisy:torch.Tensor) -> torch.Tensor:
    noisy = noisy * 32768.0
    noisy = noisy.clip(-32768.0, 32767.0)
    return noisy

def normalize(wavform:torch.Tensor) -> torch.Tensor:
    wavform = wavform.to(dtype=torch.float32)
    wavform = wavform / 32768.0
    return wavform

def load_weight(
    args:argparse.Namespace, hubert:Wav2Vec2Model, clsf:AudioClassifier, mode='origin', metaInfo:CorruptionMeta=None
) -> None:
    assert mode in ['origin', 'adaptation'], 'No support'
    if mode == 'origin':
        h_p = os.path.join(args.orig_wght_pth, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}.pt')
        c_p = os.path.join(args.orig_wght_pth, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}.pt')
    elif mode == 'adaptation':
        h_p = os.path.join(args.adpt_wght_path, f'hubert-{args.model_level}-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
        c_p = os.path.join(args.adpt_wght_path, f'clsModel-{args.model_level}-{constants.dataset_dic[args.dataset]}-{metaInfo.type}-{metaInfo.level}.pt')
    hubert.load_state_dict(state_dict=torch.load(h_p, weights_only=True))
    clsf.load_state_dict(state_dict=torch.load(c_p, weights_only=True))

def inference(args:argparse.Namespace, hubert:nn.Module, clsModel:nn.Module, data_loader:DataLoader):
    hubert.eval(); clsModel.eval()
    ttl_corr = 0.; ttl_size = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, _ = clsModel(hubert(features)[0])
        ttl_size += labels.shape[0]
        _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
    return ttl_corr / ttl_size

def build_model(args:argparse.Namespace, pre_weight:bool=True) -> tuple[torchaudio.models.Wav2Vec2Model, AudioClassifier]:
    bundle = torchaudio.pipelines.HUBERT_BASE
    if pre_weight:
        hubert = bundle.get_model().to(device=args.device)
    else:
        hubert = torchaudio.models.hubert_base().to(device=args.device)
    cfg = ConfigDict()
    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = args.class_num
    cfg.classifier['extend_size'] = 2048
    cfg.classifier['convergent_size'] = 256
    cfg.embedding = ConfigDict()
    cfg.embedding.embed_size = bundle._params['encoder_embed_dim']
    classifier = AudioClassifier(config=cfg).to(device=args.device)
    return hubert, classifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='SpeechCommandsV2', choices=['SpeechCommandsV2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_file_name', type=str, default='analysis.csv')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--orig_wght_pth', type=str)
    ap.add_argument('--adpt_wght_path', type=str)
    ap.add_argument('--use_pre_trained_weigth', action='store_true')
    ap.add_argument('--model_level', type=str, default='base', choices=['base', 'large', 'x-large'])

    args = ap.parse_args()
    if args.dataset == 'SpeechCommandsV2':
        args.class_num = 35
        args.sample_rate = 16000
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.arch = 'HuBERT'
    args.output_path = os.path.join(args.output_path, args.dataset, args.arch, 'Analysis')
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################

    corruption_types=['WHN', 'ENQ', 'END1', 'END2', 'ENSC', 'PSH', 'TST']
    corruption_levels=['L1', 'L2']
    records = pd.DataFrame(columns=['Dataset',  'Algorithm', 'Param No.', 'Corruption', 'Non-adapted', 'Adapted', 'Improved'])
    corruption_metas = corruption_meta(corruption_types=corruption_types, corruption_levels=corruption_levels)
    hubert, clsf = build_model(args=args, pre_weight=args.use_pre_trained_weigth)
    load_weight(args=args, hubert=hubert, clsf=clsf, mode='origin')
    param_no = count_ttl_params(hubert) + count_ttl_params(clsf)

    for idx, cmeta in enumerate(corruption_metas):
        print(f'{idx+1}/{len(corruption_metas)}: {args.dataset} {cmeta.type}-{cmeta.level} analyzing...')
        sc2_c = SpeechCommandsV2C(
            root_path=args.dataset_root_path, corruption_level=cmeta.level, corruption_type=cmeta.type,
        )

        org_set = TransferDataset(
            dataset=sc2_c, data_tf=ReduceChannel()
        )
        adpt_set = TransferDataset(
            dataset=sc2_c,
            data_tf=Components(transforms=[
                RNNoiseTransform(sample_rate=args.sample_rate, normalize=normalize, denormalize=denormalize),
                ReduceChannel()
            ])
        )

        org_loader = DataLoader(
            dataset=org_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )
        adpt_loader = DataLoader(
            dataset=adpt_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
            num_workers=args.num_workers
        )

        print('Origin evaluation')
        org_accu = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=org_loader)
        print('Adaptation evaluation')
        adpt_accu = inference(args=args, hubert=hubert, clsModel=clsf, data_loader=adpt_loader)
        print(f'Accuracy comparision: origin is: {org_accu:.4f}, adaptation is: {adpt_accu:.4f}')

    print('END!')