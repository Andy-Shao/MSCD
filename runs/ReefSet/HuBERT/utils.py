import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import copy

import torch
from torch import nn
import torchaudio
from torch.utils.data import DataLoader

from lib.utils import ConfigDict
from AuT.lib.model import AudioClassifier

def mlt_inference(
    args:argparse.Namespace, corruption_types:list[str], hub:nn.Module, clsf:nn.Module, 
    data_loader:DataLoader
) -> tuple[float, dict[str, float]]:
    hub.eval(); clsf.eval()
    y_ts, y_ss = {k:[] for k in corruption_types}, {k:[] for k in corruption_types}
    local_roc_aucs = {}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)
            with torch.inference_mode():
                outputs, _ = clsf(hub(features)[0])
                outputs = outputs.detach().cpu()
            y_ss[corruption_type].append(nn.functional.softmax(outputs.detach().cpu(), dim=1))
            y_ts[corruption_type].append(copy.deepcopy(labels))
    for corruption_type in corruption_types:
        y_s = torch.concat(y_ss[corruption_type], dim=0)
        y_t = torch.concat(y_ts[corruption_type], dim=0)
        local_roc_aucs[corruption_type] = roc_auc_score(y_true=y_t.numpy(), y_score=y_s.numpy(), average='macro', multi_class='ovr')
        y_ss[corruption_type] = y_s
        y_ts[corruption_type] = y_t
    y_s = torch.concat([v for k,v in y_ss.items()], dim=0)
    y_t = torch.concat([v for k,v in y_ts.items()], dim=0)
    global_roc_auc = roc_auc_score(
        y_true=y_t.numpy(), y_score=y_s.numpy(), average='macro', multi_class='ovr'
    )
    return global_roc_auc, local_roc_aucs

def teach_inference(
    args:argparse.Namespace, corruption_types:list[str], hubs:list[nn.Module], clsfs:list[nn.Module],
    data_loader:DataLoader
) -> dict[str, float]:
    for hub in hubs: hub.eval()
    for clsf in clsfs: clsf.eval()
    y_trues, y_scores = [], {it:[] for it in corruption_types}
    roc_aucs = {}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            corruption_type = corruption_types[i]
            features = data[i].to(args.device)
            hub = hubs[i]
            clsf = clsfs[i]

            with torch.inference_mode():
                outputs, _ = clsf(hub(features)[0])
                outputs = outputs.detach().cpu()
            y_scores[corruption_type].append(nn.functional.softmax(outputs.detach().cpu(), dim=1))
        y_trues.append(labels)
    y_t = torch.concat(y_trues, dim=0)
    for corruption_type in corruption_types:
        y_s = torch.concat(y_scores[corruption_type], dim=0)
        roc_aucs[corruption_type] = roc_auc_score(y_true=y_t.numpy(), y_score=y_s.numpy(), average='macro', multi_class='ovr')
    return roc_aucs

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