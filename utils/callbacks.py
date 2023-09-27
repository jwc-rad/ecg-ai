import json
import os
import numbers
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Union

import torch
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


# save results callback
class ResultsCallback(pl.Callback):
    '''Save results to SAVE_DIR
    '''
    def __init__(
        self,
        result_dir: str = './results',
        label_key='key',
        label_save_key='FILENAME',
        label_save_name='AGE',
        csv_name='valid',
    ):
        super().__init__()
        #if not isinstance(result_dir, list):
        #    result_dir = [result_dir]
        self.result_dir = result_dir
        self.label_key = label_key
        self.label_save_key = label_save_key
        self.label_save_name = label_save_name
        #self.postprocess = postprocess
        
        self.dataframe = pd.DataFrame(columns=[label_save_key, label_save_name])
        os.makedirs(result_dir, exist_ok=True)
        self.dataframe_path = os.path.join(result_dir, f'{csv_name}.csv')
        self.dataframe_new = True
    
    def _result_batch(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.dataframe = pd.read_csv(self.dataframe_path)
        
        keys = batch['metadata'][self.label_key]
        outs = pl_module.outputs[...,-1].detach().cpu().numpy()
        
        add_df = pd.DataFrame(data={
            self.label_save_key: list(keys), 
            self.label_save_name: list(outs),
        })                    
        
        self.dataframe = pd.concat([self.dataframe, add_df])
        self.dataframe.to_csv(self.dataframe_path, index=False) 
             
    
    # Callback method
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.dataframe_new:
            self.dataframe.to_csv(self.dataframe_path, index=False)   
            self.dataframe_new = False  
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.dataframe_new:
            self.dataframe.to_csv(self.dataframe_path, index=False)   
            self.dataframe_new = False  
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
