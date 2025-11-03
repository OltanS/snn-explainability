from typing import Type, Union
import torch.nn as nn

from spikingjelly.activation_based.model.sew_resnet import sew_resnet34, SEWResNet
from models.spiking_vgg import custom_spiking_vgg11, custom_spiking_vgg11_bn, SpikingVGG
from models.sew_resnet import ResNetWrapper
from models.simple_snn import SimpleSNN

model_to_name = {
    custom_spiking_vgg11: "spiking_vgg11",
    custom_spiking_vgg11_bn: "spiking_vgg11_bn",
    sew_resnet34: "sew_resnet34",
    SimpleSNN: "simple_snn",
}

name_to_model = {v: k for k, v in model_to_name.items()}

def get_model(model_name, num_classes, in_channels, spiking_neuron, pretrained=True, **kwargs):
    ModelClass: Type[nn.Module] = get_model_class(model_name)

    if "simple_snn" in model_name.lower():
        model = ModelClass(
            num_classes=num_classes,
            in_channels=in_channels,
            spiking_neuron=spiking_neuron,
            **kwargs
        )

    elif "resnet" in model_name:
        base_model = ModelClass(num_classes=num_classes, pretrained=pretrained, spiking_neuron=spiking_neuron, **kwargs)
        model = ResNetWrapper(model=base_model, in_channels=in_channels)
    
    elif "vgg" in model_name:
        model = ModelClass(
            num_classes=num_classes,
            in_channels=in_channels,
            spiking_neuron=spiking_neuron,
            pretrained=pretrained,
            **kwargs
        )

    else:
        raise ValueError(f"Model {model_name} is not implemented.")

    return model


def get_model_name(model):
    model_name = model_to_name.get(type(model), None)
    if model_name is None:
        raise ValueError(f"Model {type(model)} is not implemented.")
    return model_name

def get_model_class(model_name):
    model_class = name_to_model.get(model_name, None)
    if model_class is None:
        raise ValueError(f"Model {model_name} is not implemented.")
    return model_class