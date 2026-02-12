import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
import math


def _voting_size(num_classes, s0=20, s1=400):
    """Compute voting size for VotingLayer based on number of classes."""
    if num_classes <= s0:
        return 10
    elif num_classes >= s1:
        return 1
    else:
        return math.ceil(1 + 9 * (s1 - num_classes) / (s1 - s0))


class SimpleSNNComposite(nn.Module):
    def __init__(self, num_classes, in_channels, spiking_neuron=None, readout="direct", **kwargs):
        super().__init__()
        channels = 128
        conv = []
        for i in range(5):
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**kwargs))
            conv.append(layer.MaxPool2d(2, 2))
            in_channels = channels

        # Build readout layers
        if readout == "voting":
            v = _voting_size(num_classes)
            final_layers = [
                layer.Linear(128, num_classes * v),
                spiking_neuron(**kwargs),
                layer.VotingLayer(voting_size=v),
            ]
        elif readout == "spiking":
            final_layers = [
                layer.Linear(128, num_classes),
                spiking_neuron(**kwargs),
            ]
        else:  # direct
            final_layers = [
                layer.Linear(128, num_classes),
                nn.Identity(),
            ]

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 8, 512),
            spiking_neuron(**kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, 128),
            spiking_neuron(**kwargs),
            *final_layers,
        )
    
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)