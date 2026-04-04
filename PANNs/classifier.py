import torch
from torch import nn

from lib.utils import ConfigDict

class PannClassifier(nn.Module):
    def __init__(self, config:ConfigDict):
        super().__init__()
        self.final_classify = nn.Linear(in_features=config.embed_num, out_features=config.class_num)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.final_classify(x)
        return x