import glob
import importlib
import json
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import random
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from monai.transforms import Compose

from mislight.utils.hydra import instantiate_list

class ECGBase(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.prepare_transforms(transform)
        
    def __len__(self):        
        return self.image_size
    
    def __getitem__(self, index):
        read_items = self.read_data(index)

        return_items = self.run_transform(read_items)
        
        return return_items
    
    ## override this to define transforms
    def prepare_transforms(self, transform):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm) 
    
    ## override this to define self.keys, paths, and etc.
    def prepare_data(self):
        pass
        
    ## override this to read data by index
    def read_data(self, index):
        read_items = {}
        metadata = {}
        image_path = self.image_paths[index % self.image_size]
        image_key = self.image_keys[index % self.image_size]
        read_items['image'] = image_path
        metadata['image_path'] = image_path
        metadata['key'] = image_key
                        
        if hasattr(self, 'label_list'):
            this_label = self.label_list[index % self.image_size]
            read_items['label'] = np.array([this_label], dtype=float)
                        
        read_items['metadata'] = metadata
        return read_items    
    
    
class ECG_V0(ECGBase):
    def __init__(
        self, transform, image_dir, label_csv=None, phase='train',
        **prepare_data_kwargs,
    ):
        """
        image_dir: channel_name: path_to_dir
        """
        super().__init__(transform)
        self.image_dir = image_dir
        if label_csv is not None and os.path.exists(label_csv):
            self.label_csv = pd.read_csv(label_csv)
        else:
            self.label_csv = None
        self.phase = phase
        
        self.prepare_data(**prepare_data_kwargs)
        
    def prepare_data(self, split_path=None, cv_fold=0, image_extension='npy', label_key='id', label_name='label', parse_str=''):
        if self.phase in ['train', 'valid']:
            assert os.path.exists(split_path)
            with open(split_path, 'rb') as f:
                split = pickle.load(f)
            this_split = split[cv_fold][self.phase]
                        
            _paths = sorted(glob.glob(os.path.join(self.image_dir, f'*.{image_extension}')))
            _keys = [os.path.basename(x).split(f'.{image_extension}')[0] for x in _paths]
            
            if isinstance(parse_str, str):
                parse_str = [parse_str]
            _keys = [x for x in _keys if np.any([y in x for y in parse_str])]
            
            self.image_keys = sorted(set(_keys).intersection(set(this_split)))
            self.image_paths = [os.path.join(self.image_dir, f'{x}.{image_extension}') for x in self.image_keys]
            self.image_size = len(self.image_paths)
            
            if getattr(self, 'label_csv', None) is not None:
                label_dict = {k:v for k,v in zip(self.label_csv[label_key], self.label_csv[label_name])}
                self.label_list = [label_dict[k] for k in self.image_keys]
            
        else:
            _paths = sorted(glob.glob(os.path.join(self.image_dir, f'*.{image_extension}')))
            _keys = [os.path.basename(x).split(f'.{image_extension}')[0] for x in _paths]
            
            if not isinstance(parse_str, list):
                parse_str = [parse_str]
            _keys = [x for x in _keys if np.any([y in x for y in parse_str])]
            
            self.image_keys = _keys
            self.image_paths = [os.path.join(self.image_dir, f'{x}.{image_extension}') for x in self.image_keys]
            self.image_size = len(self.image_paths)