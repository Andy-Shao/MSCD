import os
import pandas as pd

from torch import nn
from torch.utils.data import Dataset
import torchaudio

class UrbanSound8KC(Dataset):
    meta_file = os.path.join('metadata', 'UrbanSound8K.csv')
    def __init__(
        self, root_path:str, corruption_type:str|list[str], corruption_level:str, data_tf:nn.Module|list[nn.Module]=None, 
        label_tf:nn.Module=None
    ):
        super().__init__()
        self.root_path = root_path
        if isinstance(corruption_type, str): self.corruption_types = [corruption_type]
        else: self.corruption_types = corruption_type
        self.corruption_level = corruption_level
        if data_tf is None: self.data_tfs = None
        elif isinstance(data_tf, nn.Module): self.data_tfs = [data_tf]
        else: self.data_tfs = data_tf
        self.label_tf = label_tf
        self.data_ls = pd.read_csv(os.path.join(root_path, self.meta_file), header=0)

    def __len__(self):
        return len(self.data_ls)
    
    def __getitem__(self, index):
        meta_info = self.data_ls.iloc[index]
        result = []
        for corruption_type in self.corruption_types:
            wavform, sample_rate = torchaudio.load(
                uri=os.path.join(
                    self.root_path, 'audio', corruption_type, self.corruption_level, f'fold{meta_info['fold']}', 
                    meta_info['slice_file_name']
                ), normalize=True
            )
            result.append(wavform)
        label = int(meta_info['classID'])
        if self.data_tfs is not None:
            tmp = []
            for i, wavform in enumerate(result):
                data_tf = self.data_tfs[i]
                wavform = data_tf(wavform)
                tmp.append(wavform)
            result = tmp
        if self.label_tf is not None:
            label = self.label_tf(label)
        result.append(label)
        return tuple(result)