import os
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio

class VocalSoundC(Dataset):
    labe_dic_file = 'class_labels_indices_vs.csv'
    meta_file = os.path.join('datafiles', 'te.json')
    def __init__(self, root_path:str, corruption_level:str, corruption_type:str|list[str], data_tf:nn.Module|list[nn.Module]=None, label_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        self.corruption_level = corruption_level
        if isinstance(corruption_type, str): self.corruption_types = [corruption_type]
        else: self.corruption_types = corruption_type
        if data_tf is None: self.data_tfs = None
        elif isinstance(data_tf, nn.Module): self.data_tfs = [data_tf]
        else: self.data_tfs = data_tf
        self.label_tf = label_tf
        self.label_dict = pd.read_csv(os.path.join(root_path, self.labe_dic_file), header=0)
        self.sample_list = self.__file_list__()
    
    def __file_list__(self):
        import json
        with open(os.path.join(self.root_path, self.meta_file)) as f:
            json_str = json.load(f)
        file_list = pd.json_normalize(json_str['data'])
        file_list['file_name'] = [str(it).split('/')[-1] for it in file_list['wav']]
        return file_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        meta_info = self.sample_list.iloc[index]
        label = int(pd.Series(self.label_dict[self.label_dict['mid'] == meta_info['labels']]['index']).item())
        result = []
        for corruption_type in self.corruption_types:
            wavform, sample_rate = torchaudio.load(
                os.path.join(self.root_path, 'audio_16k', corruption_type, self.corruption_level, meta_info['file_name']),
                normalize=True
            )
            result.append(wavform)
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

class SpeechCommandsV2(Dataset):
    label_dict = {
        'backward': 0, 'bed': 3, 'bird': 32, 'cat': 17, 'dog': 10, 'down': 14, 'eight': 13, 'five': 2, 
        'follow': 1, 'forward': 16, 'four': 20, 'go': 33, 'happy': 31, 'house': 8, 'learn': 6, 'left': 26, 
        'marvin': 27, 'nine': 23, 'no': 22, 'off': 24, 'on': 5, 'one': 34, 'right': 18, 'seven': 12, 
        'sheila': 30, 'six': 15, 'stop': 11, 'three': 25, 'tree': 9, 'two': 7, 'up': 29, 'visual': 19, 
        'wow': 21, 'yes': 28, 'zero': 4
    }
    def __init__(
            self, root_path:str, folder_in_archive:str='speech_commands_v0.02', mode:str=None, download:bool = True, 
            data_tf:torch.nn.Module=None, label_tf:torch.nn.Module=None
        ):
        from torchaudio.datasets import SPEECHCOMMANDS
        super(SpeechCommandsV2, self).__init__()
        assert mode in ['training', 'validation', 'testing', None], 'No support'

        if not os.path.exists(root_path):
            os.makedirs(root_path)  

        self.dataset = SPEECHCOMMANDS(
            root=root_path, url='speech_commands_v0.02', folder_in_archive=folder_in_archive, subset=mode, download=download
        )
        self.data_tf = data_tf
        self.label_tf = label_tf

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        wavform, sample_rate, label, speaker_id, utterance_num = self.dataset[index]
        label = int(self.label_dict[label])
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        return wavform, label

class SpeechCommandsV2C(Dataset):
    meta_file = 'testing_list.txt'
    def __init__(self, root_path:str, corruption_level:str, corruption_type:str|list[str], data_tf:nn.Module|list[nn.Module]=None, label_tf:nn.Module=None):
        super().__init__()
        self.root_path = root_path
        if isinstance(corruption_type, str):
            self.corruption_types = [corruption_type]
        else: self.corruption_types = corruption_type
        self.corruption_level = corruption_level
        if data_tf is None: self.data_tfs = None
        elif isinstance(data_tf, nn.Module): self.data_tfs = [data_tf]
        else: self.data_tfs = data_tf
        self.label_tf = label_tf
        self.data_ls = self.__cal_data_list__()

    def __cal_data_list__(self):
        with open(os.path.join(self.root_path, SpeechCommandsV2C.meta_file), 'r') as f:
            data_list = f.readlines()
        return [it.strip() for it in data_list]

    def __len__(self):
        return len(self.data_ls)
    
    def __getitem__(self, index):
        meta_info = self.data_ls[index]
        label = int(SpeechCommandsV2.label_dict[meta_info.split('/')[0]])
        ret = []
        for corruption_type in self.corruption_types:
            wavform, sample_rate = torchaudio.load(
                os.path.join(self.root_path, corruption_type, self.corruption_level, meta_info), 
                normalize=True
            )
            ret.append(wavform)
        if self.data_tfs is not None:
            tmp=[]
            for idx, wavform in enumerate(ret):
                data_tf = self.data_tfs[idx]
                wavform = data_tf(wavform)
                tmp.append(wavform)
            ret = tmp
        if self.label_tf is not None:
            label = self.label_tf(label)
        ret.append(label)
        return tuple(ret)