import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer
from spikingjelly.activation_based.model.spiking_vgg import (
    SpikingVGG, model_urls, cfgs, load_state_dict_from_url
)


class CustomSpikingVGG(SpikingVGG):
    """SpikingVGG with custom input channels."""
    
    def __init__(self, cfg, batch_norm=False, norm_layer=None, num_classes=1000, 
                 init_weights=True, spiking_neuron=None, in_channels=3, **kwargs):
        self.in_channels = in_channels
        super().__init__(cfg, batch_norm, norm_layer, num_classes, init_weights, 
                        spiking_neuron, **kwargs)

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
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
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
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

        # Load weights, instantiating incompatible first conv when in_channels doesn't match
        if in_channels != 3:
            # drop pretrained first conv weights
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('features.0.weight')}

            # initialize the model's first conv to appropriate random init
            first_layer = model.features[0]
            if hasattr(first_layer, 'weight') and first_layer.weight is not None:
                nn.init.kaiming_normal_(first_layer.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(first_layer, 'bias') and first_layer.bias is not None:
                nn.init.constant_(first_layer.bias, 0)

        # ignore / reinitialize last classifier layer if num_classes doesn't match
        last_idx = len(model.classifier) - 1
        if model.classifier[last_idx].out_features != 1000:
            # drop pretrained final classifier weights/bias
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith(f'classifier.{last_idx}.weight') and 
                            not k.startswith(f'classifier.{last_idx}.bias')}

            # reinitialize final classifier layer
            final_layer = model.classifier[last_idx]
            if hasattr(final_layer, 'weight') and final_layer.weight is not None:
                nn.init.normal_(final_layer.weight, 0, 0.01)
            if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, 0)

        model.load_state_dict(state_dict, strict=False)

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