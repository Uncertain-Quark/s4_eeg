import mne

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import random
import pickle
import numpy as np

class TUEG_Dataset(Dataset):
    def __init__(self, tueg_signal_filepath: str=None, 
                 window_signal: int=60,
                 Fs: int=250, n_files: int=None, n_windows: int=None,
                 norm_data: bool=True):
        seed_set=42
        random.seed(seed_set)

        super().__init__()
        with open(tueg_signal_filepath, 'r') as f:
            tueg_signal_files = f.readlines()
        
        self.tueg_signal_files = [tueg_file.strip() for tueg_file in tueg_signal_files]
        if n_files is not None:
            self.tueg_signal_files = self.tueg_signal_files[:n_files]

        self.window_signal = window_signal
        self.Fs = Fs
        self.norm_data = norm_data

        if n_windows is None:
            self.time_window = self.Fs * self.window_signal
        else:
            self.time_window = n_windows

    def __len__(self):
        return len(self.tueg_signal_files)
    
    def __getitem__(self, index):
        eeg_signal_path = self.tueg_signal_files[index]

        with open(eeg_signal_path, 'rb') as f:
            data_dict = pickle.load(f)

        data = data_dict['data']

        time_index = random.randint(0, data.shape[1] - self.time_window)
        index_data =  data[:, time_index:time_index + self.time_window]
        if self.norm_data:
            index_data = self.__normalize__(index_data)
        
        return torch.tensor(index_data).float()
    
    def __normalize__(self, eeg_data, eps: float=1e-10):
        '''Function to normalize the input data'''
        mean = np.mean(eeg_data, axis=1, keepdims=True)
        std = np.std(eeg_data, axis=1, keepdims=True)

        normalized_eeg_data = (eeg_data - mean) / (std + eps)

        return normalized_eeg_data

class TUEG_Dataset_chunked(TUEG_Dataset):
    def __init__(self, tueg_signal_filepath: str = None, window_signal: int = 60, Fs: int = 250, n_files: int = None, n_windows: int = None, norm_data: bool = True):
        super().__init__(tueg_signal_filepath, window_signal, Fs, n_files, n_windows, norm_data)
    
    def __getitem__(self, index):
        eeg_signal_path = self.tueg_signal_files[index]

        with open(eeg_signal_path, 'rb') as f:
            data_dict = pickle.load(f)

        data = data_dict['data']

        if self.norm_data:
            data = self.__normalize__(data.copy())
        
        return torch.tensor(data).float()
    
class TUEG_Dataset_chunked_psd(TUEG_Dataset):
    def __init__(self, tueg_signal_filepath: str = None, window_signal: int = 60, Fs: int = 250, n_files: int = None, n_windows: int = None, norm_data: bool = True):
        super().__init__(tueg_signal_filepath, window_signal, Fs, n_files, n_windows, norm_data)
    
    def __getitem__(self, index):
        eeg_signal_path = self.tueg_signal_files[index]

        with open(eeg_signal_path, 'rb') as f:
            data_dict = pickle.load(f)

        data = data_dict['data']
        psd = data_dict['freq_bands_np']
        psd = np.array(psd)
        psd = psd.reshape(-1, psd.shape[-1]).T
        
        return torch.tensor(data).float(), torch.tensor(psd).float()

class TUEG_Dataset_chunked_channel_select(TUEG_Dataset):
    def __init__(self, tueg_signal_filepath: str = None, window_signal: int = 60, Fs: int = 250, n_files: int = None, n_windows: int = None, norm_data: bool = True, channel_list: list=['FZ', 'CZ', 'PZ']):
        super().__init__(tueg_signal_filepath, window_signal, Fs, n_files, n_windows, norm_data)
        self.ch_names = ['C3',
                    'C4',
                    'CZ',
                    'F3',
                    'F4',
                    'F7',
                    'F8',
                    'FP1',
                    'FP2',
                    'FZ',
                    'O1',
                    'O2',
                    'P3',
                    'P4',
                    'PZ',
                    'T3',
                    'T4',
                    'T5',
                    'T6']
        self.desired_indices = [self.ch_names.index(ch) for ch in channel_list]
                        
    def __getitem__(self, index):

        eeg_signal_path = self.tueg_signal_files[index]

        with open(eeg_signal_path, 'rb') as f:
            data_dict = pickle.load(f)

        data = data_dict['data']
        data = data[self.desired_indices, :]

        if self.norm_data:
            data = self.__normalize__(data.copy())

        return torch.tensor(data).float()

        



        
