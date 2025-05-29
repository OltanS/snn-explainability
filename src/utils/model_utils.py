from models.basicsnn import BasicSNN
from models.spiking_vgg11 import SpikingVGG11

model_to_name = {
    BasicSNN: "basic_snn",
    SpikingVGG11: "vgg11"
}

name_to_model = {v: k for k, v in model_to_name.items()}

def get_model(model_name, num_classes, time_steps=30, num_channels=3, beta=0.95):
    ModelClass = get_model_class(model_name)
    return ModelClass(num_classes=num_classes, time_steps=time_steps, num_channels=num_channels, beta=beta)

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