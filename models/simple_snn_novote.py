import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
import math

class SimpleSNNNoVote(nn.Module):
    def __init__(self, num_classes, in_channels, spiking_neuron=None, **kwargs):
        super().__init__()
        channels = 128
        conv = []
        for i in range(5):
            conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
            conv.append(layer.BatchNorm2d(channels))
            conv.append(spiking_neuron(**kwargs))
            conv.append(layer.MaxPool2d(2, 2))
            in_channels = channels

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, 128),
            spiking_neuron(**kwargs),
            layer.Linear(128, num_classes),
            spiking_neuron(**kwargs),
        )
    
    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)

