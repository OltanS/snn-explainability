from ...models.basicsnn import BasicSNN
from ...models.spiking_vgg11 import SpikingVGG11

model_to_name = {
    BasicSNN: "basic_snn",
    SpikingVGG11: "vgg11"
}

name_to_model = {v: k for k, v in model_to_name.items()}

def get_model(model_name, num_classes, time_steps=30, num_channels=3):
    ModelClass = name_to_model.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Model {model_name} is not implemented.")
    
    return ModelClass(num_classes=num_classes, time_steps=time_steps, num_channels=num_channels)