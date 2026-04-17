import os
import pandas as pd
from dataclasses import dataclass

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

class UrbanSound8K(Dataset):
    @dataclass
    class USMeta:
        slice_file_name:str
        fsID:int
        start:float
        end:float
        salience:int
        fold:int
        classID:int
        classDes:str
    meta_file = os.path.join('metadata', 'UrbanSound8K.csv')
    def __init__(
        self, root_path, folds:list[int], sample_rate:int=-1, data_tf:nn.Module=None, label_tf:nn.Module=None, 
        include_rate:bool=False
    ):
        super().__init__()
        self.root_path = root_path
        self.sample_rate = sample_rate
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.include_rate = include_rate
        self.data_list = self.__cal_data_list__(folds=folds)

    def __cal_data_list__(self, folds:list[int]) -> list[USMeta]:
        meta_info = pd.read_csv(os.path.join(self.root_path, self.meta_file), header=0)
        for idx, fold in enumerate(folds):
            if idx == 0:
                select_meta_info = meta_info[meta_info['fold'] == fold]
            else:
                select_meta_info = pd.concat([select_meta_info, meta_info[meta_info['fold']==fold]], axis=0)
        ret = []
        for idx, row in select_meta_info.iterrows():
            meta = UrbanSound8K.USMeta(
                slice_file_name=row['slice_file_name'], fsID=row['fsID'], start=row['start'], end=row['end'],
                salience=row['salience'], fold=row['fold'], classID=row['classID'], classDes=row['class']
            )
            ret.append(meta)
        return ret
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        from torchaudio.transforms import Resample
        meta = self.data_list[index]
        label = int(meta.classID)
        audio_path = os.path.join(self.root_path, 'audio', f'fold{meta.fold}', meta.slice_file_name)
        wavform, sample_rate = torchaudio.load(audio_path, normalize=True)
        if self.sample_rate != -1 and sample_rate != self.sample_rate: 
            resample_ops = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            wavform = resample_ops(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.include_rate:
            return wavform, label, sample_rate if self.sample_rate == -1 else self.sample_rate
        else:
            return wavform, label