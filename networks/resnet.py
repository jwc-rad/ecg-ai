from __future__ import annotations

from typing import Any

import torch
from torch import nn

from monai.networks.nets.resnet import ResNet as monaiResNet
from monai.networks.nets.resnet import ResNetBlock, ResNetBottleneck

def get_inplanes():
    return [64, 128, 256, 512]

def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]

## ResNet with adding self.features
class ResNet(monaiResNet):
    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
class ResNet18(ResNet):
    def __init__(
        self,
        block=ResNetBlock,
        layers=[2,2,2,2],
        block_inplanes=[64, 128, 256, 512],
        #spatial_dims: int = 3,
        #n_input_channels: int = 3,
        #num_classes: int = 400,
        **kwargs,
    ) -> None:
        super().__init__(
            block,
            layers,
            block_inplanes,
            **kwargs,
        )
        
class ResNet34(ResNet):
    def __init__(
        self,
        block=ResNetBlock,
        layers=[3,4,6,3],
        block_inplanes=[64, 128, 256, 512],
        #spatial_dims: int = 3,
        #n_input_channels: int = 3,
        #num_classes: int = 400,
        **kwargs,
    ) -> None:
        super().__init__(
            block,
            layers,
            block_inplanes,
            **kwargs,
        )
    
class ResNet50(ResNet):
    def __init__(
        self,
        block=ResNetBottleneck,
        layers=[3,4,6,3],
        block_inplanes=[64, 128, 256, 512],
        #spatial_dims: int = 3,
        #n_input_channels: int = 3,
        #num_classes: int = 400,
        **kwargs,
    ) -> None:
        super().__init__(
            block,
            layers,
            block_inplanes,
            **kwargs,
        )