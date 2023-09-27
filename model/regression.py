import itertools
import numpy as np

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import wandb

from monai.networks.utils import one_hot

from mislight.networks.utils import load_pretrained_net
from mislight.models import BaseModel
from mislight.models.utils import instantiate_scheduler

class RegressionModel(BaseModel):
    def __init__(self, **opt):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.is_train = opt['train']
        self.use_wandb = 'wandb' in opt['logger']
        
        # define networks
        if self.is_train:
            self.net_names = ['netA']
        else:
            self.net_names = ['netA']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                net = getattr(self, net)
                net = load_pretrained_net(net, pretrained)
        
        # define loss functions
        if self.is_train:            
            self.criterion = instantiate(opt['loss'], _convert_='partial')

        # define inferer
        if 'inferer' in opt:
            self.inferer = instantiate(opt['inferer'], _convert_='partial')
            
        # define metrics
        if 'metrics' in opt:
            self.metrics = {}
            if opt['metrics'] is not None:
                for k, v in opt['metrics'].items():
                    if '_target_' in v:
                        self.metrics[k] = instantiate(v, _convert_='partial')

    ### custom methods
    def set_input(self, batch):
        self.image = batch['image']
        if 'label' in batch.keys():
            self.label = batch['label']
    
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.image_out = self.forward(self.image)

    ### pl methods
    
    def configure_optimizers(self):        
        netparams = [getattr(self, n).parameters() for n in ['netA'] if hasattr(self, n)]
        optimizer_GF = instantiate(self.hparams['optimizer'], params=itertools.chain(*netparams))
        
        optimizers = [optimizer_GF]
        schedulers = [{
            k: instantiate_scheduler(optimizer, v) if k=='scheduler' else v 
            for k,v in self.hparams['scheduler'].items()
        } for optimizer in optimizers]
        
        return optimizers, schedulers
    
    def forward(self, x):
        out = self.netA(x)
        return out
                    
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        bs = self.image.size(0)
                                    
        # Classification loss: S(A) ~ Ya
        w0 = self.hparams['lambda']
        if w0 > 0:            
            loss_C = self.criterion(self.image_out, self.label)
            #self.log('loss/cls', loss_C, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_C = 0
        loss += loss_C * w0  
        
        # combine loss
        self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        stage = 'valid'
        self._step_forward(batch, batch_idx)
        self.outputs = self.image_out
        
        bs = self.image.size(0)  
        
        #loss = None
        loss = self.criterion(self.outputs, self.label)
        
        if hasattr(self, 'metrics'):
            met_outputs = self.outputs
            #met_outputs = (torch.sigmoid(self.image_cls) > 0.5)
            for k in self.metrics.keys():
                self.metrics[k](met_outputs.float(), self.label.float())
                if self.global_step == 0 and self.use_wandb:
                    wandb.define_metric(f'metrics/valid_{k}', summary='min')
        
        self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)
                    
        return loss
    
    def on_validation_epoch_end(self):
        if not hasattr(self, 'metrics'):
            return
        for k in self.metrics.keys():
            if self.metrics[k].get_buffer() is not None:
                mean_metric = self.metrics[k].aggregate()
                if isinstance(mean_metric, list):
                    mean_metric = mean_metric[0]
                mean_metric = mean_metric.item()
                self.metrics[k].reset()
                self.log(f'metrics/valid_{k}', mean_metric)
    
    def predict_step(self, batch, batch_idx):
        self.set_input(batch)
        self.outputs = self.forward(self.image)
        return None
    
    def test_step(self, batch, batch_idx):
        self.set_input(batch)
        self.outputs = self.forward(self.image)
        return None


        

    