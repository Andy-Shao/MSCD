import argparse
from tqdm import tqdm
import os

import torch 
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.models import Wav2Vec2Model

from lib import constants
from lib.utils import ConfigDict
from lib.corruption import CorruptionMeta
from AuT.lib.model import AudioClassifier

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