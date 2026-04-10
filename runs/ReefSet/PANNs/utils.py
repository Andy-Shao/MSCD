import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import DataLoader

from lib.utils import ConfigDict
from PANNs.models import Wavegram_Logmel_Cnn14
from PANNs.classifier import PANClassifier

def build_model(args:argparse.Namespace, use_pre_weight:bool=True) -> tuple[Wavegram_Logmel_Cnn14, PANClassifier]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pan = Wavegram_Logmel_Cnn14(
        sample_rate=32000, 
        window_size=1024,
        hop_size=320,
        mel_bins=64, 
        fmin=50,
        fmax=14000,
        classes_num=527 #audio set
    )
    if use_pre_weight:
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
    clsf = PANClassifier(config=cfg)

    pan, clsf = pan.to(device=args.device), clsf.to(device=args.device)
    return pan, clsf

def inference(
    args:argparse.Namespace, pan:Wavegram_Logmel_Cnn14, clsf:PANClassifier, data_loader:DataLoader
) -> float:
    pan.eval(); clsf.eval()
    for i, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features = features.to(args.device)

        with torch.inference_mode():
            outputs = clsf(pan(features)['embedding'])
            outputs = outputs.detach().cpu()
        if i == 0:
            y_t = [labels.detach()]
            y_s = [nn.functional.softmax(outputs, dim=1)]
        else:
            y_t.append(labels.detach())
            y_s.append(nn.functional.softmax(outputs, dim=1))
    eval_roc_auc = roc_auc_score(
        y_true=torch.concat(y_t, dim=0).numpy(), y_score=torch.concat(y_s, dim=0).numpy(),
        average='macro', multi_class='ovr'
    )
    return eval_roc_auc