from typing import Any

import torch 
from torch import nn

class FCEmbedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, width=128, num_layers=[6,8]) -> None:
        super(FCEmbedding, self).__init__()
        ng = 32
        self.restnet = FCEResNet(cin=num_channels, width=width, ng=ng, num_layers=num_layers)
        self.patch_embedding = nn.Conv1d(in_channels=width*(2**(1+len(num_layers))), out_channels=embed_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x:torch.Tensor):
        x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)

        return x

class FCEResNet(nn.Module):
    def __init__(self, cin:int, width:int, ng:int, num_layers:list[int]):
        super(FCEResNet, self).__init__()
        self.root = nn.Sequential(
            StdConv1d(in_channels=cin, out_channels=width, kernel_size=14, stride=2, bias=False, padding=6),
            nn.BatchNorm1d(num_features=width, eps=1e-6),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool1d(kernel_size=6, stride=2, padding=2)

        w = width
        self.layers = nn.ModuleList()
        for i, layer_num in enumerate(num_layers):
            if i == 0:
                self.layers.append(FCEResNetBlock(cin=w, cout=w*4, cmid=w, ng=ng))
                for _ in range(layer_num): self.layers.append(FCEResNetBlock(cin=w*4, cout=w*4, cmid=w, ng=ng))
                w = w * 4
            else:
                self.layers.append(FCEResNetBlock(cin=w, cout=w*2, cmid=w//2, stride=2, ng=ng))
                for _ in range(layer_num): self.layers.append(FCEResNetBlock(cin=w*2, cout=w*2, cmid=w//2, ng=ng))
                w = w * 2

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layers: x = l(x)
        return x

class FCEResNetBlock(nn.Module):
    def __init__(self, cin:int, cout:int, cmid:int, ng:int, stride=1) -> None:
        super(FCEResNetBlock, self).__init__()

        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.norm1 = nn.BatchNorm1d(num_features=cmid, eps=1e-6)
        
        self.conv2 = conv7(cmid, cmid, stride=stride, bias=False)
        self.norm2 = nn.BatchNorm1d(num_features=cmid, eps=1e-6)

        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.norm3 = nn.BatchNorm1d(num_features=cout, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride=stride, bias=False)
            self.ds_gn = nn.BatchNorm1d(num_features=cout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.ds_gn(residual)

        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.norm3(self.conv3(y))

        y = self.relu(residual + y)
        return y

def conv1x1(cin:int, cout:int, stride=1, bias=False) -> nn.Conv1d:
    return StdConv1d(
        in_channels=cin, out_channels=cout, kernel_size=1, stride=stride, padding=0, bias=bias
    )

def conv7(cin:int, cout:int, stride=1, groups=1, bias=False) -> nn.Conv1d:
    return StdConv1d(
        in_channels=cin, out_channels=cout, kernel_size=7, stride=stride, padding=3, bias=bias, 
        groups=groups
    )

class StdConv1d(nn.Conv1d):
    def forward(self, x) -> Any:
        import torch.nn.functional as F

        w = self.weight
        var, mean = torch.var_mean(w, dim=[1, 2], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-5)
        return F.conv1d(
            input=x, weight=w, bias=self.bias, stride=self.stride, padding=self.padding, 
            dilation=self.dilation, groups=self.groups
        )