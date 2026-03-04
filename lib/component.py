import numpy as np
import random

import torch
from torch import nn

class MelSpectrogramPadding(nn.Module):
    def __init__(self, target_length):
        super(MelSpectrogramPadding, self).__init__()
        self.target_length = target_length

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        p = self.target_length - x.shape[2]
        if p > 0:
            x = pad(x, (0, p, 0, 0), mode='constant', value=0.)
        elif p < 0:
            x = x[:, :, 0:self.target_length]
        return x

class TimeShift(nn.Module):
    def __init__(self, shift_limit: float, is_random=True, is_bidirection=False) -> None:
        """
        Time shift data augmentation

        :param shift_limit: shift_limit -> (-1, 1), shift_limit < 0 is left shift
        """
        super().__init__()
        self.shift_limit = shift_limit
        self.is_random = is_random
        self.is_bidirection = is_bidirection

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        if self.is_random:
            shift_arg = int(random.random() * self.shift_limit * wavform.shape[1])
            if self.is_bidirection:
                shift_arg = int((random.random() * 2 - 1) * self.shift_limit * wavform.shape[1])
        else:
            shift_arg = int(self.shift_limit * wavform.shape[1])
        return wavform.roll(shifts=shift_arg)

class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x

class AudioPadding(nn.Module):
    def __init__(self, max_length:int, sample_rate:int, random_shift:bool=False):
        super(AudioPadding, self).__init__()
        self.max_length = max_length
        self.random_shift = random_shift

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        l = self.max_length - x.shape[1]
        if l > 0:
            if self.random_shift:
                head = random.randint(0, l)
                tail = l - head
            else:
                head = l // 2
                tail = l - head
            x = pad(x, (head, tail), mode='constant', value=0.)
        return x

class Components(nn.Module):
    def __init__(self, transforms: list) -> None:
        super().__init__()
        assert transforms is not None, 'No support'
        self.transforms = nn.ModuleList(transforms)

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            wavform = transform(wavform)
        return wavform

# class RNNoiseTransform(nn.Module):
#     def __init__(self, sample_rate:int, normalize, denormalize):
#         super().__init__()
#         self.sample_rate = sample_rate
#         self.normalize = normalize
#         self.denormalize = denormalize

#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         import pyrnnoise

#         denoiser = pyrnnoise.RNNoise(self.sample_rate)
#         x = self.denormalize(x)
#         x = x.numpy().astype(np.int16)
#         y = [denoise_audio for speech_prob, denoise_audio in denoiser.denoise_chunk(x, partial=True)]
#         y = torch.tensor(np.concat(y, axis=1))
#         y = self.normalize(y)
#         return y

class ReduceChannel(nn.Module):
    def __init__(self):
        super(ReduceChannel, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.squeeze(x, dim=0)

class AmplitudeToDB(nn.Module):
    def __init__(self, top_db:float, max_out:float) -> None:
        from torchaudio import transforms
        super(AmplitudeToDB, self).__init__()
        self.model = transforms.AmplitudeToDB(top_db=top_db)
        self.max_out = max_out
        self.top_db = top_db

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x) / (self.top_db // self.max_out)
    
class FrequenceTokenTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        c, token_num, token_len = x.size()
        x = x.reshape(-1, token_len)
        return x

class AudioClip(nn.Module):
    def __init__(self, max_length:int, mode:str='head', is_random:bool=False):
        super(AudioClip, self).__init__()
        assert mode in ['head', 'mid', 'tail']
        self.max_length = max_length
        self.mode = mode
        self.is_random = is_random

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        l = x.shape[1] - self.max_length
        if l > 0:
            if self.is_random:
                start = np.random.randint(low=0, high=l)
            elif self.mode == 'head':
                start = 0
            elif self.mode == 'mid':
                start = int(l/2.)
            elif self.mode == 'tail':
                start = l
            x = x[:, start:start+self.max_length]
        return x
    
class OneHot2Index(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> int:
        _, pred = torch.max(x, dim=0)
        return pred.item()