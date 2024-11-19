import torch
from torch import nn


def modify_last_layer(model, model_name, num_classes):
 
    if model_name.startswith("resnet"):  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.startswith("mobilenet_v3"):  
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} is not supported for modification.")
    return model