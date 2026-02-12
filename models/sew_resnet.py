import math
import torch
import torch.nn as nn

from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.sew_resnet import SEWResNet


def _voting_size(num_classes, s0=20, s1=400):
    """Compute voting size for VotingLayer based on number of classes."""
    if num_classes <= s0:
        return 10
    elif num_classes >= s1:
        return 1
    else:
        return math.ceil(1 + 9 * (s1 - num_classes) / (s1 - s0))


class ResNetWrapper(nn.Module):
    """SEWResNet with custom input channels and configurable readout."""
    
    #TODO: Add dropout
    def __init__(self, model: SEWResNet, spiking_neuron=None, in_channels=3, readout="direct", **kwargs):
        super().__init__()
        self.model = model
        self.model.conv1 = layer.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        if readout == "spiking":
            self.readout_layer = spiking_neuron(**kwargs) if spiking_neuron is not None else nn.Identity()
        elif readout == "voting" and spiking_neuron is not None:
            num_classes = model.fc.out_features
            v = _voting_size(num_classes)
            self.model.fc = layer.Linear(self.model.fc.in_features, num_classes * v)
            self.readout_layer = nn.Sequential(
                spiking_neuron(**kwargs),
                layer.VotingLayer(voting_size=v),
            )
        else:  # direct
            self.readout_layer = nn.Identity()
    
    def forward(self, x):
        x = self.model(x)
        x = self.readout_layer(x)
        return x
    
