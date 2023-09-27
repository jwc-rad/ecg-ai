import torch
from torch import nn

from monai.networks.nets.efficientnet import EfficientNetBN


class EfficientNetBNFeatures(EfficientNetBN):
    def features(self, inputs: torch.Tensor):        
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        
        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))
        return x

    def forward(self, inputs: torch.Tensor):
        x = self.features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)

        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

class EfficientNetBNFeaturesNoHead(EfficientNetBN):
    def features(self, inputs: torch.Tensor):        
        # Stem
        x = self._conv_stem(self._conv_stem_padding(inputs))
        x = self._swish(self._bn0(x))
        # Blocks
        x = self._blocks(x)
        
        return x

    def forward(self, inputs: torch.Tensor):
        x = self.features(inputs)

        # Head
        x = self._conv_head(self._conv_head_padding(x))
        x = self._swish(self._bn1(x))

        # Pooling and final linear layer
        x = self._avg_pooling(x)

        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x
