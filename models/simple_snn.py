import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
import math

class SimpleSNN(nn.Module):
    def __init__(self, num_classes, in_channels, spiking_neuron=None, have_last_spiking_neuron=True, **kwargs):
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
            spiking_neuron(**kwargs) if have_last_spiking_neuron else nn.Identity(),
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