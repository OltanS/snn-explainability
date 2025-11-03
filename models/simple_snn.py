import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
import math

class SimpleSNN(nn.Module):
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

        voting_size, total_outputs = SimpleSNN._voting_and_total(num_classes)

        self.conv_fc = nn.Sequential(
            *conv,

            layer.Flatten(),
            layer.Dropout(0.5),
            layer.Linear(channels * 4 * 4, 512),
            spiking_neuron(**kwargs),

            layer.Dropout(0.5),
            layer.Linear(512, total_outputs),
            spiking_neuron(**kwargs),

            layer.VotingLayer(voting_size=voting_size)
        )

    def _voting_and_total(num_classes, s0=128, s1=512):
        if num_classes <= s0:
            v = 10
        elif num_classes >= s1:
            v = 1
        else:
            v = math.ceil(1 + 9 * (s1 - num_classes) / (s1 - s0))
        return v, num_classes * v

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)
