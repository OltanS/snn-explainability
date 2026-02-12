from typing import Type
from functools import partial

import torch.nn as nn

from spikingjelly.activation_based.layer import ThresholdDependentBatchNorm2d
from spikingjelly.activation_based.model.sew_resnet import sew_resnet34
from spikingjelly.activation_based import layer, neuron

from models.spiking_vgg import custom_spiking_vgg11, custom_spiking_vgg11_bn, custom_spiking_vgg16_bn
from models.sew_resnet import ResNetWrapper
from models.simple_snn import SimpleSNN
from models.simple_snn_composite import SimpleSNNComposite

model_to_name = {
    custom_spiking_vgg11: "spiking_vgg11",
    custom_spiking_vgg11_bn: "spiking_vgg11_bn",
    custom_spiking_vgg16_bn: "spiking_vgg16_bn",
    sew_resnet34: "sew_resnet34",
    SimpleSNN: "simple_snn",
    SimpleSNNComposite: "simple_snn_composite",
}

name_to_model = {v: k for k, v in model_to_name.items()}
AVAILABLE_MODELS = list(name_to_model.keys())

def get_model(model_name, num_classes, in_channels, spiking_neuron, pretrained=False, readout="direct", **kwargs):
    ModelClass: Type[nn.Module] = get_model_class(model_name)

    norm_layer = partial(ThresholdDependentBatchNorm2d, alpha=1.0, v_th=1.0)

    if "simple_snn" in model_name.lower():
        model = ModelClass(
            num_classes=num_classes,
            in_channels=in_channels,
            spiking_neuron=spiking_neuron,
            readout=readout,
            **kwargs
        )

    elif "resnet" in model_name:
        def _filter_wrapper_kwargs(kwargs):
            exclude = {'pretrained', 'cnf', 'spiking_neuron', 'readout'}
            return {k: v for k, v in kwargs.items() if k not in exclude}
        wrapper_kwargs = _filter_wrapper_kwargs(kwargs)

        base_model = ModelClass(
            num_classes=num_classes,
            pretrained=pretrained,
            cnf="ADD",
            spiking_neuron=spiking_neuron,
            norm_layer=norm_layer,
            **kwargs
        )
        
        if "spiking" not in model_name:
            spiking_neuron = None

        model = ResNetWrapper(model=base_model, spiking_neuron=spiking_neuron, in_channels=in_channels, readout=readout, **wrapper_kwargs)
    
    elif "vgg" in model_name:
        model = ModelClass(
            num_classes=num_classes,
            in_channels=in_channels,
            spiking_neuron=spiking_neuron,
            pretrained=pretrained,
            norm_layer=norm_layer,
            readout=readout,
            **kwargs
        )

    else:
        raise ValueError(f"Model {model_name} is not implemented.")

    # Add the model name as an attribute for easy retrieval
    model._model_name = model_name

    return model


def get_model_name(model):
    # Retrieve the name from the stored attribute
    if hasattr(model, '_model_name'):
        return model._model_name
    else:
        raise ValueError(f"Model {type(model)} does not have a stored name.")

def get_model_class(model_name):
    model_class = name_to_model.get(model_name, None)
    if model_class is None:
        raise ValueError(f"Model {model_name} is not implemented.")
    return model_class

def get_conv_layers(model):
    return [m for m in model.modules() if isinstance(m, layer.Conv2d)]

def get_lif_layers(model):
    return [m for m in model.modules() if isinstance(m, (neuron.LIFNode, neuron.IFNode))]