import torch
import torch.nn as nn

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.sew_resnet import SEWResNet

class ResNetWrapper(nn.Module):
    """SEWResNet with custom input channels."""
    
    def __init__(self, model: SEWResNet, in_channels=3):
        super().__init__()
        self.model = model
        self.model.conv1 = layer.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        return self.model(x)
