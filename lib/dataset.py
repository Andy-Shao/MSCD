import os
import pandas as pd
import shutil
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from lib.utils import one_hot_to_hex, hex_to_one_hot

def mlt_load_from(
    root_path:str, index_file_name:str, data_tfs:list[nn.Module]=None, label_tf=None,
    is_one_hot_label:bool=False, class_num:int=-1
) -> Dataset:
    class MltLoadDs(Dataset):
        def __init__(self):
            super().__init__()
            self.data_index = pd.read_csv(os.path.join(root_path, index_file_name), index_col=0)
            self.length = self.data_index.shape[0]

        def __len__(self):
            return self.length
        
        def __getitem__(self, index):
            row = self.data_index.iloc[index]
            ret = []
            for i in range(row.shape[0] - 1):
                feature = np.load(os.path.join(root_path, row.iloc[i]))
                feature = torch.from_numpy(feature)
                if data_tfs is not None:
                    feature = data_tfs[i](feature)
                ret.append(feature)
            label = int(row['label']) if not is_one_hot_label else hex_to_one_hot(hex_str=row['label'], length=class_num)
            if label_tf is not None:
                label = label_tf(label)
            ret.append(label)
            return tuple(ret)
    return MltLoadDs()

def mlt_store_to(
    dataset:Dataset, root_path:str, index_file_name:str, data_tfs:list[nn.Module], label_tf:nn.Module=None,
    is_one_hot_label:bool=False
) -> None:
    print(f'Store dataset into {root_path}, meta file is: {index_file_name}')
    columns = []
    for i in range(len(data_tfs)):
        columns.append(f'data_path{i}')
    columns.append('label')
    data_index = pd.DataFrame(columns=columns)
    try: 
        if os.path.exists(root_path): shutil.rmtree(root_path)
        os.makedirs(root_path)
    except:
        raise Exception('remove director has an error')
    for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
        label = data[-1] if not is_one_hot_label else one_hot_to_hex(data[-1])
        if label_tf is not None:
            label = label_tf(label)
        row = []
        for i in range(len(data_tfs)):
            feature = data[i]
            if data_tfs[i] is not None:
                feature = data_tfs[i](feature)
            data_path = f'{idx}-{i}_{label}.npy'
            np.save(file=os.path.join(root_path, data_path), arr=feature.detach().numpy())
            row.append(data_path)
        row.append(label)
        data_index.loc[len(data_index)] = row
    data_index.to_csv(os.path.join(root_path, index_file_name))

class GpuMultiTFDataset(Dataset):
    def __init__(self, dataset:Dataset, tfs:list[nn.Module], device:str='cuda', maintain_cpu:bool=True):
        super(GpuMultiTFDataset, self).__init__()
        assert tfs is not None, 'No support'
        self.dataset = dataset
        self.tfs = [tf.to(device) for tf in tfs]
        self.device = device
        self.maintain_cpu = maintain_cpu

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item, label = self.dataset[index]
        item = item.to(self.device)
        ret = [item.clone() for _ in range(len(self.tfs))]
        for i, tf in enumerate(self.tfs):
            if tf is not None:
                ret[i] = tf(ret[i])
                if self.maintain_cpu:
                    ret[i] = ret[i].to('cpu')
        ret.append(label)
        return tuple(ret)

class Subset(Dataset):
    def __init__(self, dataset: Dataset, label_list:list[int]):
        super().__init__()
        self.dataset = dataset
        self.label_list = label_list
    
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return self.dataset[self.label_list[index]]


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
        
class IdxSet(Dataset):
    def __init__(self, dataset:Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        ret = [index] + list(data)
        return tuple(ret)
    
class PseudoLabelSet(Dataset):
    def __init__(self, dataset:Dataset, pseudo_labels:dict[int, torch.Tensor], label_position:int=1):
        super().__init__()
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels
        self.label_position = label_position

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        import copy
        data = self.dataset[index]
        data = list(data)
        pseudo_label = self.pseudo_labels[index]
        pseudo_label = copy.deepcopy(pseudo_label)
        data[self.label_position] = pseudo_label
        return tuple(data)