import copy
import math
import numpy as np
import torch

# Sources: 
# https://github.com/bneyshabur/generalization-bounds
# https://github.com/nitarshan/robust-generalization-measures
# Fantastic Generalization Measures and Where to Find Them: arXiv:1912.02178

# Reparametrization
def reparametrize_model(model, previous_layer = None):
    """
    Reparametrize the model by adjusting BatchNorm parameters and updating
    weights and biases of preceding layers.

    Args:
        model (nn.Module): The PyTorch model to reparametrize.

    Returns:
        nn.Module: The reparametrized model.
    """
    # Recursively iterate over children modules
    for child in model.children():
        module_name = child.__class__.__name__

        # Recursively reparametrize child layers
        previous_layer = reparametrize_model(child, previous_layer)

        # Track previous layers for Linear/Conv layers
        if module_name in {"Linear", "Conv1d", "Conv2d", "Conv3d"}:
            previous_layer = child

        # Handle BatchNorm layers and update parameters
        elif module_name in {"BatchNorm1d", "BatchNorm2d"}:
            if previous_layer is None:
                raise ValueError(f"BatchNorm layer {child} must follow a Linear or Conv layer.")
            _update_batchnorm_parameters(batchnorm_layer=child, preceding_layer=previous_layer)

    return previous_layer


def calculate_margin(model, device, val_loader):
    """
    Calculate the margin for a model on a validation dataset.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        device (torch.device): Device (CPU or GPU) for computation.
        val_loader (torch.utils.data.DataLoader): Validation data loader.

    Returns:
        float: The 5th percentile of margins.
    """
    margins = []
    model.eval()

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            correct_logits = output[torch.arange(len(target)), target]
            max_other_logits, _ = output.masked_fill(
                torch.nn.functional.one_hot(target, output.size(1)).bool(), float("-inf")
            ).max(dim=1)

            margins.extend((correct_logits - max_other_logits).tolist())

    val_margin = np.percentile(margins, 5)
    return val_margin


def _update_batchnorm_parameters(batchnorm_layer, preceding_layer):
    """
    Update the BatchNorm layer parameters and adjust weights/biases of the preceding layer.

    Args:
        batchnorm_layer (nn.Module): The BatchNorm layer to update.
        preceding_layer (nn.Module): The preceding Linear or Conv layer.
    """
    with torch.no_grad():
        # Compute scaling factors for weights
        scale = batchnorm_layer.weight / ((batchnorm_layer.running_var + batchnorm_layer.eps).sqrt())

        # Update biases in the preceding layer
        preceding_layer.bias.copy_(batchnorm_layer.bias + scale * (preceding_layer.bias - batchnorm_layer.running_mean))

        # Update weights in the preceding layer
        perm = list(reversed(range(preceding_layer.weight.dim())))
        preceding_layer.weight.copy_((preceding_layer.weight.permute(perm) * scale).permute(perm))

        # Reset the BatchNorm layer parameters
        batchnorm_layer.bias.zero_()
        batchnorm_layer.weight.fill_(1)
        batchnorm_layer.running_mean.zero_()
        batchnorm_layer.running_var.fill_(1)


def _reset_batchnorm_parameters(batchnorm_layer):
    """
    Reset BatchNorm layer parameters to default values.

    Args:
        batchnorm_layer (nn.Module): The BatchNorm layer to reset.
    """
    with torch.no_grad():
        batchnorm_layer.bias.zero_()
        batchnorm_layer.weight.fill_(1)
        batchnorm_layer.running_mean.zero_()
        batchnorm_layer.running_var.fill_(1)


# Measure Calculation
def calculate_measure(model, init_model, measure_func, operator, kwargs=None, p=1.0):
    """
    Recursively calculates measures based on the specified operator ('product', 'norm', etc.)
    across model layers.
    """
    kwargs = kwargs or {}
    if operator == "product":
        return math.exp(calculate_measure(model, init_model, measure_func, "log_product", kwargs, p))
    elif operator == "norm":
        return calculate_measure(model, init_model, measure_func, "sum", kwargs, p) ** (1 / p)

    measure_value = 0
    for child, init_child in zip(model.children(), init_model.children()):
        module_name = child._get_name()
        if module_name in ["Linear", "Conv1d", "Conv2d", "Conv3d"]:
            if operator == "log_product":
                measure_value += math.log(measure_func(child, init_child, **kwargs))
            elif operator == "sum":
                measure_value += measure_func(child, init_child, **kwargs) ** p
            elif operator == "max":
                measure_value = max(measure_value, measure_func(child, init_child, **kwargs))
        else:
            measure_value += calculate_measure(child, init_child, measure_func, operator, kwargs, p)
    return measure_value


# Utility Functions
def calculate_norm(module, p=2.0, q=2.0):
    """Calculates the specified norm of a module's weight."""
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()


def calculate_operator_norm(module, p=float("Inf")):
    """Calculates the operator norm of a module's weight."""
    _, singular_values, _ = module.weight.view(module.weight.size(0), -1).svd()
    return singular_values.norm(p).item()


def calculate_distance(module, init_module, p=2, q=2):
    """Calculates the distance between a module's current and initial weights."""
    return (module.weight - init_module.weight).view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()


def calculate_hidden_distance(module, init_module, p=2, q=2):
    """Calculates the hidden-layer-weighted distance."""
    return (get_hidden_units(module) ** (1 - 1 / q)) * calculate_distance(module, init_module, p, q)


def calculate_hidden_operator_norm(module, init_module, p=2, q=2, p_op=float("Inf")):
    """Calculates the hidden-layer-weighted operator norm."""
    return calculate_hidden_distance(module, init_module, p, q) / calculate_operator_norm(module, p_op)


def get_hidden_units(module):
    """Returns the number of hidden units in a module."""
    return module.weight.size(0)


def get_num_parameters(module):
    """Calculates the number of parameters in a module."""
    bias_param = 0 if module.bias is None else module.bias.size(0)
    return bias_param + module.weight.size(0) * module.weight.view(module.weight.size(0), -1).size(1)


def calculate_path_norm(model, device, p=2.0, input_size=(3, 64, 64)):
    """Calculates the Lp path norm of the model."""
    tmp_model = copy.deepcopy(model)
    print([child for child in tmp_model.children()])
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones((1, *input_size)).to(device)
    return tmp_model.forward(data_ones).sum().item() ** (1 / p)



# Main Calculation
def calculate_generalization_bounds(trained_model, init_model, train_loader, val_loader, nchannels, img_dim, device="cpu"):
    """
    Calculates various generalization bounds and measures for a model.
    """

    margin = calculate_margin(trained_model, device, val_loader)

    model = copy.deepcopy(trained_model)
    init_model = copy.deepcopy(init_model)

    reparametrize_model(model)
    reparametrize_model(init_model)

    num_samples = len(train_loader.dataset)
    depth = calculate_measure(model, init_model, measure_func=get_hidden_units, operator="sum")
    num_parameters = calculate_measure(model, init_model, measure_func=get_num_parameters, operator="sum")

    measures, bounds = {}, {}
    with torch.no_grad():
        # Norm-based Measures
        norm_settings = {"model": model, "init_model": init_model, "measure_func": calculate_norm, "operator": "product"}

        measures["L_{1,inf} norm"] = calculate_measure(**norm_settings, kwargs={"p": 1, "q": np.inf}) / margin
        measures["Frobenius norm"] = calculate_measure(**norm_settings, kwargs={"p": 2, "q": 2}) / margin
        measures["L_{3,1.5} norm"] = calculate_measure(**norm_settings, kwargs={"p": 3, "q": 1.5}) / margin

        opperator_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": calculate_operator_norm,
            "operator": "product",
        }

        measures["Spectral norm"] = calculate_measure(**opperator_settings, kwargs={"p": float("Inf")}) / margin
        measures["L_1.5 operator norm"] = calculate_measure(**opperator_settings, kwargs={"p": 1.5}) / margin
        measures["Trace norm"] = calculate_measure(**opperator_settings, kwargs={"p": 1}) / margin

        # Enhanced Norm Metrics
        log_product_settings = {
            "model": model,
            "init_model": init_model,
            "operator": "log_product",
        }
        measures["Log Product of Spectral Norms"] = (
            calculate_measure(measure_func=calculate_operator_norm, **log_product_settings, kwargs={"p": float("Inf")})
            / margin
        )
        measures["Log Product of Frobenius Norms"] = (
            calculate_measure(measure_func=calculate_norm, **log_product_settings, kwargs={"p": 2, "q": 2}) / margin
        )
        measures["Frobenius over Spectral Norm"] = measures["Frobenius norm"] / measures["Spectral norm"]

        # Distance Metrics
        distance_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": calculate_distance,
            "operator": "sum",
        }
        measures["Frobenius Distance"] = calculate_measure(**distance_settings, kwargs={"p": 2, "q": 2}) / margin
        measures["Spectral Distance"] = calculate_measure(**distance_settings, kwargs={"p": float("Inf")}) / margin

        # Path Norms
        input_size = (nchannels, img_dim, img_dim)
        measures["L1_path norm"] = calculate_path_norm(model, device, p=1, input_size=input_size) / margin
        measures["L1.5_path norm"] = calculate_path_norm(model, device, p=1.5, input_size=input_size) / margin
        measures["L2_path norm"] = calculate_path_norm(model, device, p=2, input_size=input_size) / margin

        # Bound Calculations
        alpha = math.sqrt(depth + math.log(nchannels * img_dim**2))
        bounds["L1_max Bound"] = alpha * measures["L_{1,inf} norm"] / math.sqrt(num_samples)
        bounds["Frobenius Bound"] = alpha * measures["Frobenius norm"] / math.sqrt(num_samples)
        bounds["L_{3,1.5} Bound"] = alpha * measures["L_{3,1.5} norm"] / (num_samples ** (1 / 3))

        # Enhanced Bounds
        beta = math.log(num_samples) * math.log(num_parameters)
        ratio_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": calculate_hidden_operator_norm,
            "operator": "norm",
        }
        ratio = calculate_measure(**ratio_settings, p=2 / 3, kwargs={"p": 2, "q": 1, "p_op": np.inf})
        bounds["Spec_L2_1 Bound"] = beta * measures["Spectral norm"] * ratio / math.sqrt(num_samples)
        ratio = calculate_measure(**ratio_settings, kwargs={"p": 2, "q": 2, "p_op": np.inf}, p=2)
        bounds["Spec_Fro Bound"] = beta * measures["Spectral norm"] * ratio / math.sqrt(num_parameters)

        # Flatness-Based Bounds
        sigma = calculate_path_norm(model, device, p=2, input_size=input_size)
        bounds["Flatness"] = 1 / (sigma**2)

    return measures, bounds
