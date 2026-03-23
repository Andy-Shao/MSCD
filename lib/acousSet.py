import os
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio

class ReefSet(Dataset):
    def __init__(
            self, root_path:str, mode:str, include_rate:bool=False, data_tf:nn.Module=None, label_tf:nn.Module=None,
            train_annotations_file:str='./data/ReefSet/train_annotations.csv', 
            test_annotations_file:str='./data/ReefSet/test_annotations.csv', label_mode:str='single'
        ):
        super().__init__()
        assert label_mode in ['multiple', 'single'], 'No support'
        self.root_path = root_path
        assert mode in ['train', 'test'], 'No support'
        self.mode = mode
        self.include_rate = include_rate
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.meta_info = pd.read_csv(train_annotations_file) if mode == 'train' else pd.read_csv(test_annotations_file)
        self.label_dic = ReefSetC.__label_dic__(label_mode)

    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, index):
        meta = self.meta_info.iloc[index]
        wavform, sample_rate = torchaudio.load(os.path.join(self.root_path, 'full_dataset', meta['file_name']), normalize=True)
        eye_matrix = torch.eye(len(self.label_dic), dtype=float)
        label = torch.zeros_like(eye_matrix[0], dtype=float)
        for k in self.label_dic[meta['label']]:
            label += eye_matrix[k]
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        if self.include_rate: 
            return wavform, label, sample_rate
        else:
            return wavform, label

class ReefSetC(Dataset):
    meta_file = 'test_annotations.csv'
    def __init__(
        self, root_path:str, corruption_type:str|list[str], corruption_level:str, data_tf:nn.Module|list[nn.Module]=None, label_tf:nn.Module=None, 
        label_mode:str='single'):
        super().__init__()
        assert label_mode in ['multiple', 'single']
        self.root_path = root_path
        if isinstance(data_tf, nn.Module): self.data_tfs = [data_tf]
        else: self.data_tfs = data_tf
        self.label_tf = label_tf
        self.meta_infos = pd.read_csv(os.path.join(root_path, self.meta_file), header=0)
        self.label_dic = ReefSetC.__label_dic__(label_mode)
        if isinstance(corruption_type, str): self.corruption_types = [corruption_type]
        else: self.corruption_types = corruption_type
        self.corruption_level = corruption_level

    def __len__(self):
        return len(self.meta_infos)

    def __getitem__(self, index):
        meta = self.meta_infos.iloc[index]
        ret = []
        for corruption_type in self.corruption_types:
            wavform, sample_rate = torchaudio.load(
                uri=os.path.join(self.root_path, corruption_type, self.corruption_level, meta['file_name']), normalize=True
            )
            ret.append(wavform)
        eye_matrix = torch.eye(len(self.label_dic), dtype=float)
        label = torch.zeros_like(eye_matrix[0], dtype=float)
        for k in self.label_dic[meta['label']]:
            label += eye_matrix[k]
        if self.data_tfs is not None:
            tmp = []
            for idx, wavform in enumerate(ret):
                data_tf = self.data_tfs[idx]
                tmp.append(data_tf(wavform))
            ret = tmp
        if self.label_tf is not None:
            label = self.label_tf(label)
        ret.append(label)
        return tuple(ret)

    @staticmethod
    def __label_dic__(label_mode) -> dict:
        if label_mode == 'multiple':
            return {
            'ambient': [0], 

            'anthrop_boat_engine': [1],
            'anthrop_bomb': [2],
            'anthrop_mechanical': [3],

            'bioph': [4],
            'bioph_cascading_saw': [4, 5],
            'bioph_chatter': [4, 6],
            'bioph_chorus': [4, 7],
            'bioph_crackle': [4, 8],
            'bioph_croak': [4, 9],
            'bioph_damselfish': [4, 10],
            'bioph_dolphin': [4, 11],
            'bioph_double_pulse': [4, 12],
            'bioph_echinidae': [4, 13],
            'bioph_epigut': [4, 14],
            'bioph_grazing': [4, 15],
            'bioph_grouper_a': [4, 16],
            'bioph_grouper_groan': [4, 17],
            'bioph_growl': [4, 18],
            'bioph_holocentrus': [4, 19],
            'bioph_knock': [4, 20],
            'bioph_knock_croak_a': [4, 20, 21],
            'bioph_knock_croak_b': [4, 20, 22],
            'bioph_knock_croak_c': [4, 20, 23],
            'bioph_low_growl': [4, 24],
            'bioph_megnov': [4, 25],
            'bioph_midshipman': [4, 26],
            'bioph_mycbon': [4, 27],
            'bioph_pomamb': [4, 28],
            'bioph_pulse': [4, 29],
            'bioph_rattle': [4, 30],
            'bioph_rattle_response': [4, 31],
            'bioph_series_a': [4, 32],
            'bioph_series_b': [4, 33],
            'bioph_stridulation': [4, 34],
            'bioph_whup': [4, 35], 

            'geoph_waves': [36]
            }
        else:
            return {
            'ambient': [0], 

            'anthrop_boat_engine': [1],
            'anthrop_bomb': [2],
            'anthrop_mechanical': [3],

            'bioph': [4],
            'bioph_cascading_saw': [5],
            'bioph_chatter': [6],
            'bioph_chorus': [7],
            'bioph_crackle': [8],
            'bioph_croak': [9],
            'bioph_damselfish': [10],
            'bioph_dolphin': [11],
            'bioph_double_pulse': [12],
            'bioph_echinidae': [13],
            'bioph_epigut': [14],
            'bioph_grazing': [15],
            'bioph_grouper_a': [16],
            'bioph_grouper_groan': [17],
            'bioph_growl': [18],
            'bioph_holocentrus': [19],
            'bioph_knock': [20],
            'bioph_knock_croak_a': [21],
            'bioph_knock_croak_b': [22],
            'bioph_knock_croak_c': [23],
            'bioph_low_growl': [24],
            'bioph_megnov': [25],
            'bioph_midshipman': [26],
            'bioph_mycbon': [27],
            'bioph_pomamb': [28],
            'bioph_pulse': [29],
            'bioph_rattle': [30],
            'bioph_rattle_response': [31],
            'bioph_series_a': [32],
            'bioph_series_b': [33],
            'bioph_stridulation': [34],
            'bioph_whup': [35], 

            'geoph_waves': [36]
            }