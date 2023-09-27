import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

from monai.networks.layers import trunc_normal_
from monai.networks.blocks.transformerblock import TransformerBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.register_buffer('pe', pe)
        trunc_normal_(self.pe, mean=0.0, std=0.02, a=-2.0, b=2.0)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)



class CNNTransformer_ClsToken(nn.Module):
    def __init__(
        self,
        cnn_class=nn.Conv1d,
        cnn_cfg=None,
        num_classes=1,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = cnn_class(**cnn_cfg)
        
        self.positional_encoder = PositionalEncoding(hidden_size, dropout_rate)
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))        
        self.clf = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.cnn.features(x)
        
        x_shape = x.shape[2:]
        einops_dims = ' '.join([x for i,x in enumerate(['h','w','d']) if i < len(x_shape)])
        x = rearrange(x, f"b c {einops_dims} -> b ({einops_dims}) c")
        
        x = self.positional_encoder(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.clf(x[:,0])
        return x

class CNNTransformer_AvgPool(nn.Module):
    def __init__(
        self,
        cnn_class=nn.Conv1d,
        cnn_cfg=None,
        num_classes=1,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = cnn_class(**cnn_cfg)
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))     
        self.pool = nn.AdaptiveAvgPool1d(1)   
        self.clf = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.cnn.features(x)
        
        x_shape = x.shape[2:]
        einops_dims = ' '.join([x for i,x in enumerate(['h','w','d']) if i < len(x_shape)])
        x = rearrange(x, f"b c {einops_dims} -> b ({einops_dims}) c")

        #cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        #x = torch.cat((cls_token, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        x = rearrange(x, f"b ({einops_dims}) c -> b c {einops_dims}")
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.clf(x)
        return x