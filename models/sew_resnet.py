import torch
import torch.nn as nn

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.sew_resnet import SEWResNet

class ResNetWrapper(nn.Module):
    """SEWResNet with custom input channels."""
    
    #TODO: Add dropout
    def __init__(self, model: SEWResNet, spiking_neuron=None, in_channels=3, **kwargs):
        super().__init__()
        self.model = model
        self.model.conv1 = layer.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.sn = spiking_neuron(**kwargs) if spiking_neuron is not None else nn.Identity()
    
    def forward(self, x):
        x = self.model(x)
        x = self.sn(x)
        return x
    
