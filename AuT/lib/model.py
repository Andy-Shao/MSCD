import torch
from torch import nn

from lib.utils import ConfigDict

def init_weights(m: nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1 or class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1., .02)
        nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class AudioClassifier(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(AudioClassifier, self).__init__()
        embed_size = config.embedding['embed_size']
        extend_size = config.classifier['extend_size']
        convergent_size = config.classifier['convergent_size']
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc1 = nn.Linear(in_features=embed_size, out_features=extend_size, bias=True)
        # self.bn1 = nn.BatchNorm1d(num_features=extend_size, affine=True, eps=1e-6)
        self.fc2 = nn.Linear(in_features=extend_size, out_features=convergent_size)
        self.bn2 = nn.BatchNorm1d(num_features=convergent_size, affine=True, eps=1e-6)
        self.fc2.apply(init_weights)
        self.fc3 = nn.utils.parametrizations.weight_norm(
            module=nn.Linear(in_features=convergent_size, out_features=config.classifier['class_num']), name='weight')
        self.fc3.apply(init_weights)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, token_num, token_len = x.size()
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = x.contiguous().view(batch_size, token_len)

        x = self.fc1(x)
        features = self.bn2(self.fc2(x))
        outputs = self.fc3(features)

        return outputs, features