import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.models import Wav2Vec2Model

from HuBERT.lib.model import HuBClassifier

def build_model(args:argparse.Namespace, pre_weight:bool=True) -> tuple[Wav2Vec2Model, HuBClassifier]:
    bundle = torchaudio.pipelines.HUBERT_BASE
    if pre_weight:
        hubert = bundle.get_model().to(device=args.device)
    else:
        hubert = torchaudio.models.hubert_base().to(device=args.device)
    classifier = HuBClassifier(embed_size=bundle._params['encoder_embed_dim'], class_num=args.class_num, num_layers=[2]).to(device=args.device)
    return hubert, classifier

def teach_inference(
    args:argparse.Namespace, corruption_types:list[str], hubs:list[nn.Module], clsfs:list[nn.Module],
    data_loader:DataLoader,
) -> dict[str, float]:
    for hub in hubs: hub.eval()
    for clsf in clsfs: clsf.eval()
    y_trues, y_preds = [], {k:[] for k in corruption_types}
    for data in tqdm(data_loader):
        labels = data[-1]
        for i in range(len(data)-1):
            features = data[i].to(args.device)
            hub = hubs[i]; clsf = clsfs[i]
            corruption_type = corruption_types[i]
            with torch.inference_mode():
                outputs = clsf(hub(features)[0])
                outputs = outputs.detach().cpu()
            _, preds = torch.max(outputs, dim=1)
            y_preds[corruption_type].append(preds)
        y_trues.append(labels)
    y_trues = torch.concat(y_trues, dim=0)
    f1s = {}
    for corruption_type in corruption_types:
        y_pred = torch.concat(y_preds[corruption_type], dim=0)
        f1s[corruption_type] = f1_score(y_true=y_trues.numpy(), y_pred=y_pred.numpy(), average='macro')
    return f1s