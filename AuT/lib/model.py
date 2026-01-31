from torch import nn
import torch

from .embed import FCEmbedding
from lib.utils import ConfigDict

class FCEClassifier(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(FCEClassifier, self).__init__()
        embed_size = config.embedding['embed_size']

        self.merge = nn.Conv1d(in_channels=config.classifier['in_embed_num'], out_channels=1, kernel_size=15, stride=1, padding=7)
        self.norm = nn.LayerNorm(normalized_shape=embed_size, eps=1e-6)
        self.fc = nn.Linear(in_features=embed_size, out_features=config.classifier['class_num'])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.squeeze(self.merge(x), dim=1)
        x = self.norm(x)
        x = self.fc(x)

        return x

class FCETransform(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(FCETransform, self).__init__()
        embed_size = config.embedding['embed_size']
        self.embedding = FCEmbedding(
            num_channels=config.embedding['channel_num'], embed_size=embed_size,
            num_layers=config.embedding['num_layers'], width=config.embedding['width']
        )
        self.tf_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.transform['layer_num'])])
        self.drop_out = nn.Dropout(p=config.embedding['marsked_rate'])
        self.pos_embed = nn.Parameter(torch.zeros(1, config.embedding['embed_num']+2, embed_size))
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        torch.nn.init.trunc_normal_(self.cls_token, std=.02)
        self.tail_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        torch.nn.init.trunc_normal_(self.tail_token, std=.02)

    def forward(self, x:torch.Tensor):
        batch_size, token_num, token_len = x.size()
        x = self.embedding(x)
        cls_tokens = self.cls_token.repeat([batch_size, 1, 1])
        tail_tokens = self.tail_token.repeat([batch_size, 1, 1])
        x = torch.cat([cls_tokens, x, tail_tokens], dim=1)
        x = x + self.pos_embed
        x = self.drop_out(x)

        for layer in self.layers:
            x = layer(x)
        x = self.tf_norm(x)

        tokens = torch.cat([x[:, :1, :], x[:, -1:, :]], dim=1)

        return x, tokens

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

class AttentionBlock(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(AttentionBlock, self).__init__()
        embed_size = config.embedding['embed_size']
        self.attention_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.attention = MultiHeadAttention(
            d=embed_size, h=config.transform['head_num'], at_dp=config.transform['atten_drop_rate'])
        self.ffn = MultilayerPerceptron(
            fin=embed_size, fmid=config.transform['mlp_mid'], fout=embed_size,
            dp_rt=config.transform['mlp_dp_rt']
        )
        self.ffn_norm = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.attention_norm(x)
        y = self.attention(y)
        x = y + x

        y = self.ffn_norm(x)
        y = self.ffn(x)
        x = y + x
        return x

class MultilayerPerceptron(nn.Module):
    def __init__(self, fin:int, fmid:int, fout:int, dp_rt:float=.5) -> None:
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_features=fin, out_features=fmid)
        self.fc2 = nn.Linear(in_features=fmid, out_features=fout)
        self.act_fn = nn.GELU()
        self.drop_out = nn.Dropout(p=dp_rt)

        self.init_weight()

    def init_weight(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.drop_out(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d:int, h:int, at_dp:float=.5) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d % h == 0, 'it is not divisible'
        self.d = d
        self.h = h
        self.d_k, self.d_v = int(d/h), int(d/h)

        self.q = nn.Linear(in_features=d, out_features=d)
        self.k = nn.Linear(in_features=d, out_features=d)
        self.v = nn.Linear(in_features=d, out_features=d)
        self.atten_drop_rate = at_dp

        self.o = nn.Linear(in_features=d, out_features=d)

    def forward(self, x:torch.Tensor, mask=None) -> torch.Tensor:
        from torch.nn import functional as F
        batch_size, seq_length, embed_size = x.size()

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.h, self.d_v).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            query=Q, key=K, value=V, is_causal=False, dropout_p=self.atten_drop_rate
        )
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)

        return self.o(attention)