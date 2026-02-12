import math
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.spiking_vgg import (
    SpikingVGG, model_urls, cfgs, load_state_dict_from_url
)


def _voting_size(num_classes, s0=20, s1=400):
    """Compute voting size for VotingLayer based on number of classes."""
    if num_classes <= s0:
        return 10
    elif num_classes >= s1:
        return 1
    else:
        return math.ceil(1 + 9 * (s1 - num_classes) / (s1 - s0))


class CustomSpikingVGG(SpikingVGG):
    """SpikingVGG with custom input channels and configurable readout."""
    
    def __init__(self, cfg, batch_norm=False, norm_layer=None, num_classes=1000, 
                 init_weights=True, spiking_neuron=None, in_channels=3, readout="direct", **kwargs):
        self.in_channels = in_channels
        self.readout = readout
        super().__init__(cfg, batch_norm, norm_layer, num_classes, init_weights, spiking_neuron, **kwargs)

        if self.readout == "spiking":
            self.classifier = nn.Sequential(
                *list(self.classifier.children()),
                spiking_neuron(**deepcopy(kwargs))
            )
        elif self.readout == "voting":
            v = _voting_size(num_classes)
            # Replace last Linear with one outputting num_classes * v, then spiking + VotingLayer
            children = list(self.classifier.children())
            children[-1] = layer.Linear(4096, num_classes * v)
            children.append(spiking_neuron(**deepcopy(kwargs)))
            children.append(layer.VotingLayer(voting_size=v))
            self.classifier = nn.Sequential(*children)

    def make_layers(self, cfg, batch_norm=False, norm_layer=None, neuron=None, **kwargs):
        """Modified make_layers that uses custom in_channels."""
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1, padding_mode="reflect")
                if batch_norm:
                    layers += [conv2d, norm_layer(num_features=v), neuron(**deepcopy(kwargs))]
                else:
                    layers += [conv2d, neuron(**deepcopy(kwargs))]
                in_channels = v
        return nn.Sequential(*layers)

def _custom_spiking_vgg(arch, cfg, batch_norm, pretrained, progress, norm_layer=None, 
                        spiking_neuron=None, in_channels=3, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    if batch_norm:
        norm_layer = norm_layer
    else:
        norm_layer = None
    
    model = CustomSpikingVGG(
        cfg=cfgs[cfg], 
        batch_norm=batch_norm, 
        norm_layer=norm_layer, 
        spiking_neuron=spiking_neuron,
        in_channels=in_channels,
        **kwargs
    )
    
    if pretrained:
        raise NotImplementedError("Pretrained weights for custom input channels are not available, as they are meant for 3 input channels and 100+ timesteps.")

    return model

def custom_spiking_vgg11(pretrained=False, progress=True, spiking_neuron=None, 
                         in_channels=3, **kwargs):
    """Custom Spiking VGG-11 with configurable input channels."""
    return _custom_spiking_vgg('vgg11', 'A', False, pretrained, progress, None, 
                              spiking_neuron, in_channels, **kwargs)


def custom_spiking_vgg11_bn(pretrained=False, progress=True, norm_layer=None, 
                            spiking_neuron=None, in_channels=3, **kwargs):
    """Custom Spiking VGG-11-BN with configurable input channels."""
    return _custom_spiking_vgg('vgg11_bn', 'A', True, pretrained, progress, norm_layer, 
                              spiking_neuron, in_channels, **kwargs)


def custom_spiking_vgg13(pretrained=False, progress=True, spiking_neuron=None, 
                         in_channels=3, **kwargs):
    """Custom Spiking VGG-13 with configurable input channels."""
    return _custom_spiking_vgg('vgg13', 'B', False, pretrained, progress, None, 
                              spiking_neuron, in_channels, **kwargs)


def custom_spiking_vgg13_bn(pretrained=False, progress=True, norm_layer=None, 
                            spiking_neuron=None, in_channels=3, **kwargs):
    """Custom Spiking VGG-13-BN with configurable input channels."""
    return _custom_spiking_vgg('vgg13_bn', 'B', True, pretrained, progress, norm_layer, 
                              spiking_neuron, in_channels, **kwargs)


def custom_spiking_vgg16(pretrained=False, progress=True, spiking_neuron=None, 
                         in_channels=3, **kwargs):
    """Custom Spiking VGG-16 with configurable input channels."""
    return _custom_spiking_vgg('vgg16', 'D', False, pretrained, progress, None, 
                              spiking_neuron, in_channels, **kwargs)


def custom_spiking_vgg16_bn(pretrained=False, progress=True, norm_layer=None, 
                            spiking_neuron=None, in_channels=3, **kwargs):
    """Custom Spiking VGG-16-BN with configurable input channels."""
    return _custom_spiking_vgg('vgg16_bn', 'D', True, pretrained, progress, norm_layer, 
                              spiking_neuron, in_channels, **kwargs)


def custom_spiking_vgg19(pretrained=False, progress=True, spiking_neuron=None, 
                         in_channels=3, **kwargs):
    """Custom Spiking VGG-19 with configurable input channels."""
    return _custom_spiking_vgg('vgg19', 'E', False, pretrained, progress, None, 
                              spiking_neuron, in_channels, **kwargs)


def custom_spiking_vgg19_bn(pretrained=False, progress=True, norm_layer=None, 
                            spiking_neuron=None, in_channels=3, **kwargs):
    """Custom Spiking VGG-19-BN with configurable input channels."""
    return _custom_spiking_vgg('vgg19_bn', 'E', True, pretrained, progress, norm_layer, 
                              spiking_neuron, in_channels, **kwargs)