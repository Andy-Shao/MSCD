from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset

class Subset(Dataset):
    def __init__(self, dataset: Dataset, label_list:list[int]):
        super().__init__()
        self.dataset = dataset
        self.label_list = label_list
        self.set_IDs = self.__data_scanning__()

    def __data_scanning__(self) -> list[int]:
        return [idx for idx, (feature, label) in enumerate(self.dataset) if label in self.label_list]
    
    def __len__(self):
        return len(self.set_IDs)

    def __getitem__(self, index):
        return self.dataset[self.set_IDs[index]]


class TransferDataset(Dataset):
    def __init__(self, dataset: Dataset, data_tf:nn.Module=None, label_tf:nn.Module=None, device='cpu', keep_cpu=True) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_tf = data_tf if device == 'cpu' or data_tf is None else data_tf.to(device=device)
        self.label_tf = label_tf if device == 'cpu' or label_tf is None else label_tf.to(device=device)
        self.device = device
        self.keep_cpu = keep_cpu

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        feature, label = self.dataset[index]
        if self.device != 'cpu':
            feature = feature.to(self.device)
            label = label.to(self.device) if isinstance(label, torch.Tensor) else label
        if self.data_tf is not None:
            feature = self.data_tf(feature)
        if self.label_tf is not None:
            label = self.label_tf(label)

        if self.device != 'cpu' and self.keep_cpu:
            return feature.cpu(), label.cpu() if isinstance(label, torch.Tensor) else label
        else:
            return feature, label