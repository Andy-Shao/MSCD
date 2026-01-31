import torch

def denormalize(noisy:torch.Tensor) -> torch.Tensor:
    noisy = noisy * 32768.0
    noisy = noisy.clip(-32768.0, 32767.0)
    return noisy

def normalize(wavform:torch.Tensor) -> torch.Tensor:
    wavform = wavform.to(dtype=torch.float32)
    wavform = wavform / 32768.0
    return wavform