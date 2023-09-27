import torch
from torch import nn
from einops.layers.torch import Rearrange

class CRNN(nn.Module):
    def __init__(
        self,
        cnn_class=nn.Conv1d,
        cnn_cfg=None,
        rnn_class=nn.LSTM,
        rnn_cfg=None,
        clf_input_size=1,
        num_classes=1,
    ) -> None:
        super().__init__()

        self.cnn = cnn_class(**cnn_cfg)
        self.rnn = rnn_class(**rnn_cfg)
        
        self.rearrange = Rearrange("b c l -> b l c")
        
        self.clf = nn.Linear(clf_input_size, num_classes)
        
    def forward(self, x):
        x = self.cnn.features(x)
        x = self.rearrange(x)
        x, _ = self.rnn(x)
        
        x = self.clf(x[:,-1])
        return x
    
class CRNN_AvgPool(nn.Module):
    def __init__(
        self,
        cnn_class=nn.Conv1d,
        cnn_cfg=None,
        rnn_class=nn.LSTM,
        rnn_cfg=None,
        clf_input_size=1,
        num_classes=1,
    ) -> None:
        super().__init__()

        self.cnn = cnn_class(**cnn_cfg)
        self.rnn = rnn_class(**rnn_cfg)
        
        self.rearrange = Rearrange("b c l -> b l c")
        self.pool_rearrange = Rearrange("b l c -> b c l")
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clf = nn.Linear(clf_input_size, num_classes)
        
    def forward(self, x):
        x = self.cnn.features(x)
        x = self.rearrange(x)
        x, _ = self.rnn(x)
        
        x = self.pool_rearrange(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.clf(x)
        return x