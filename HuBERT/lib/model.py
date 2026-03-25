import torch 
from torch import nn

from AuT.lib.embed import StdConv1d, FCEResNetBlock

class HuBClassifier(nn.Module):
    def __init__(self, embed_size:int, class_num:int, num_layers=[6,8]):
        super().__init__()

        self.root = nn.Sequential(
            StdConv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=14, stride=2, bias=False, padding=6),
            nn.BatchNorm1d(num_features=embed_size, eps=1e-6),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool1d(kernel_size=6, stride=2, padding=2)

        w = embed_size
        self.layers = nn.ModuleList()
        for i, layer_num in enumerate(num_layers):
                self.layers.append(FCEResNetBlock(cin=w, cout=w*2, cmid=w//2, stride=2, ng=32))
                for idx in range(layer_num): 
                    if idx != layer_num - 1:
                        self.layers.append(FCEResNetBlock(cin=w*2, cout=w*2, cmid=w//2, ng=32))
                    else:
                         self.layers.append(FCEResNetBlock(cin=w*2, cout=w, cmid=w//2, ng=32))

        self.avgPool = nn.AdaptiveAvgPool1d(output_size=1)
        self.norm1 = nn.LayerNorm(normalized_shape=embed_size//2)
        self.fc1 = nn.Linear(in_features=embed_size, out_features=embed_size//2)
        self.fc2 = nn.Linear(in_features=embed_size//2, out_features=class_num)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layers:
            x = l(x)
        x = torch.squeeze(self.avgPool(x), dim=2)
        x = self.norm1(self.fc1(x))
        x = self.fc2(x)
        return x