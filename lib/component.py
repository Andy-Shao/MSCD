import numpy as np

import torch
from torch import nn

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

class RNNoiseTransform(nn.Module):
    def __init__(self, sample_rate:int, normalize, denormalize):
        super().__init__()
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.denormalize = denormalize

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        import pyrnnoise

        denoiser = pyrnnoise.RNNoise(self.sample_rate)
        x = self.denormalize(x)
        x = x.numpy().astype(np.int16)
        y = [denoise_audio for speech_prob, denoise_audio in denoiser.denoise_chunk(x, partial=True)]
        y = torch.tensor(np.concat(y, axis=1))
        y = self.normalize(y)
        return y

class ReduceChannel(nn.Module):
    def __init__(self):
        super(ReduceChannel, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.squeeze(x, dim=0)