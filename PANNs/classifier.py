import torch
from torch import nn

from lib.utils import ConfigDict

def init_weights(m: nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight, 1., .02)
        nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class PANClassifier(nn.Module):
    def __init__(self, config:ConfigDict):
        super().__init__()
        embed_num = config.embed_num
        self.fc1 = nn.Linear(in_features=embed_num, out_features=embed_num//2, bias=True)
        self.fc1.apply(init_weights)
        self.bn1 = nn.BatchNorm1d(num_features=embed_num//2, affine=True, eps=1e-6)
        self.bn1.apply(init_weights)
        self.fc2 = nn.Linear(in_features=embed_num//2, out_features=embed_num//4)
        self.fc2.apply(init_weights)
        self.bn2 = nn.BatchNorm1d(num_features=embed_num//4, affine=True, eps=1e-6)
        self.bn2.apply(init_weights)
        self.fc3 = nn.utils.parametrizations.weight_norm(
            module=nn.Linear(in_features=embed_num//4, out_features=config.class_num), name='weight')
        self.fc3.apply(init_weights)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = self.fc3(x)
        return x