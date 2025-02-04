import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Sources:
# https://github.com/bneyshabur/generalization-bounds
# https://github.com/nitarshan/robust-generalization-measures
# Fantastic Generalization Measures and Where to Find Them: arXiv:1912.02178


# Reparametrization
def reparametrize_model(model, previous_layer=None):
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


def calculate_margin(model: nn.Module, device: str, dataloader: DataLoader):
    margins = []
    model.eval()
    model.to(device)

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            correct_logits = output[torch.arange(len(target)), target]
            max_other_logits, _ = output.masked_fill(
                torch.nn.functional.one_hot(target, output.size(1)).bool(), float("-inf")
            ).max(dim=1)

            margins.extend((correct_logits - max_other_logits).tolist())
    val_margin = np.percentile(margins, 10)
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
        preceding_layer.bias.copy_(
            batchnorm_layer.bias + scale * (preceding_layer.bias - batchnorm_layer.running_mean)
        )

        # Update weights in the preceding layer
        perm = list(reversed(range(preceding_layer.weight.dim())))
        preceding_layer.weight.copy_((preceding_layer.weight.permute(perm) * scale).permute(perm))

        # Reset the BatchNorm layer parameters
        batchnorm_layer.bias.zero_()
        batchnorm_layer.weight.fill_(1)
        batchnorm_layer.running_mean.zero_()
        batchnorm_layer.running_var.fill_(1)


# Measure Calculation
def calculate_measure(model, init_model, measure_func, operator, kwargs=None, l=1.0):
    """
    Recursively calculates measures based on the specified operator ('product', 'norm', etc.)
    across model layers.
    """
    kwargs = kwargs or {}
    if operator == "product":
        return math.exp(calculate_measure(model, init_model, measure_func, "log_product", kwargs, l))
    elif operator == "norm":
        return calculate_measure(model, init_model, measure_func, "sum", kwargs, l) ** (1 / l)
    else:
        measure_value = 0
        for child, init_child in zip(model.children(), init_model.children()):
            module_name = child._get_name()
            if module_name in ["Linear", "Conv1d", "Conv2d", "Conv3d"]:
                if operator == "log_product":
                    measure_value += math.log(measure_func(child, init_child, **kwargs))
                elif operator == "sum":
                    measure_value += measure_func(child, init_child, **kwargs) ** l
                elif operator == "max":
                    measure_value = max(measure_value, measure_func(child, init_child, **kwargs))
            else:
                measure_value += calculate_measure(child, init_child, measure_func, operator, kwargs, l)
    return measure_value


# Utility Functions
def calculate_norm(module, init_module, p=2.0, q=2.0):
    """Calculates the specified norm of a module's weight."""
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()


def calculate_operator_norm(module, init_module, p=float("Inf")):
    """Calculates the operator norm of a module's weight."""
    _, singular_values, _ = module.weight.view(module.weight.size(0), -1).svd()
    return singular_values.norm(p).item()


def calculate_distance(module, init_module, p=2, q=2):
    """Calculates the distance between a module's current and initial weights."""
    return (
        (module.weight - init_module.weight).view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()
    )


def calculate_hidden_distance(module, init_module, p=2, q=2):
    """Calculates the hidden-layer-weighted distance."""
    return (get_hidden_units(module, init_module) ** (1 - 1 / q)) * calculate_distance(
        module, init_module, p, q
    )


def calculate_hidden_operator_norm(module, init_module, p=2, q=2, p_op=float("Inf")):
    """Calculates the hidden-layer-weighted operator norm."""
    return calculate_hidden_distance(module, init_module, p, q) / calculate_operator_norm(module, p_op)


def get_hidden_units(module, init_module):
    """Returns the number of hidden units in a module."""
    return module.weight.size(0)


def get_depth(module, init_module):
    return 1


def fro_over_spec(module, init_module, p=0):
    """Calculates fro norm of module (if p=0) or fro norm of distance (if p=1), over spectral norm of module"""
    spec = calculate_operator_norm(module, init_module, p=float("Inf"))
    if p == 0:
        return calculate_norm(module, init_module) / spec
    elif p == 1:
        return calculate_distance(module, init_module) / spec


def get_num_parameters(module, init_module):
    """Calculates the number of parameters in a module."""
    bias_param = 0 if module.bias is None else module.bias.size(0)
    return bias_param + module.weight.size(0) * module.weight.view(module.weight.size(0), -1).size(1)


def calculate_path_norm(model, device, p=2.0, input_size=(3, 64, 64)):
    """Calculates the Lp path norm of the model."""
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones((1, *input_size)).to(device)
    return tmp_model.forward(data_ones).sum().item() ** (1 / p)


# Main Calculation!
def calculate_generalization_bounds(
    trained_model: nn.Module,
    init_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nchannels: int = 3,
    img_dim: int = 32,
    device: str = "cpu",
):
    """
    Calculates various generalization bounds and measures for a model.
    """

    margin = calculate_margin(trained_model, device, train_loader)

    print(f"Margin: {margin}")
    model = copy.deepcopy(trained_model)
    init_model = copy.deepcopy(init_model)

    reparametrize_model(model)
    reparametrize_model(init_model)

    model.to(device)
    init_model.to(device)

    num_samples = len(train_loader.dataset)
    depth = calculate_measure(
        model, init_model, measure_func=get_depth, operator="sum"
    )  # includes output layer
    num_parameters = calculate_measure(model, init_model, measure_func=get_num_parameters, operator="sum")

    measures, bounds = {}, {}
    with torch.no_grad():
        # Norm-based Measures
        norm_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": calculate_norm,
            "operator": "product",
        }
        measures["Margin"] = margin
        measures["1 / Margin"] = 1 / margin
        measures["L_{1,inf} norm"] = calculate_measure(**norm_settings, kwargs={"p": 1, "q": np.inf})  # l=2
        measures["Frobenius norm"] = calculate_measure(**norm_settings, kwargs={"p": 2, "q": 2})  # l=2
        # measures["L_{3,1.5} norm"] = calculate_measure(**norm_settings, kwargs={"p": 3, "q": 1.5})

        opperator_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": calculate_operator_norm,
            "operator": "product",
        }
        spec_norm = calculate_measure(**opperator_settings, kwargs={"p": float("Inf")})  # l=2
        measures["Spectral norm"] = spec_norm
        # measures["L_1.5 operator norm"] = calculate_measure(**opperator_settings, kwargs={"p": 1.5})
        measures["Trace norm"] = calculate_measure(**opperator_settings, kwargs={"p": 1})

        # Norms over margin
        measures["L_{1,inf} norm over margin"] = measures["L_{1,inf} norm"] / margin
        measures["Frobenius norm over margin"] = measures["Frobenius norm"] / margin
        measures["Spectral norm over margin"] = spec_norm / margin
        measures["Trace norm over margin"] = measures["Trace norm"] / margin

        # Norms over sqared margin
        measures["L_{1,inf} norm over squared margin"] = measures["L_{1,inf} norm"] / margin**2
        measures["Frobenius norm over squared margin"] = measures["Frobenius norm"] / margin**2
        measures["Spectral norm over squared margin"] = spec_norm / margin**2
        measures["Trace norm over squared margin"] = measures["Trace norm"] / margin**2

        fraction_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": fro_over_spec,
            "operator": "sum",
        }
        measures["Mu_fro-spec"] = calculate_measure(**fraction_settings, kwargs={"p": 1})  # l=2
        measures["Mu_spec-init-main"] = (
            spec_norm * calculate_measure(**fraction_settings, kwargs={"p": 0}) / margin**2
        )  # l=2
        measures["Mu_spec-origin-main"] = spec_norm * measures["Mu_fro-spec"] / margin**2
        measures["Mu_sum-of-fro"] = depth * measures["Frobenius norm"] ** (1 / depth)
        measures["Mu_sum-of-fro/margin"] = depth * measures["Frobenius norm over squared margin"] ** (
            1 / depth
        )

        # Enhanced Norm Metrics
        log_product_settings = {
            "model": model,
            "init_model": init_model,
            "operator": "log_product",
        }
        measures["Log Product of Spectral Norms"] = calculate_measure(
            measure_func=calculate_operator_norm, **log_product_settings, kwargs={"p": float("Inf")}
        )
        measures["Log Product of Frobenius Norms"] = calculate_measure(
            measure_func=calculate_norm, **log_product_settings, kwargs={"p": 2, "q": 2}
        )

        # Distance Metrics
        distance_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": calculate_distance,
            "operator": "sum",
        }
        measures["Frobenius Distance"] = calculate_measure(
            **distance_settings, kwargs={"p": 2, "q": 2}
        )  # l=2
        measures["Spectral Distance"] = calculate_measure(
            **distance_settings, kwargs={"p": float("Inf")}
        )  # l=2

        ############################ not seen ###########################
        ################################################################
        # Path Norms
        input_size = (nchannels, img_dim, img_dim)
        measures["Mu path-norm"] = calculate_path_norm(model, device, p=2, input_size=input_size)
        measures["Mu path-norm/margin"] = measures["Mu path-norm"] / margin**2
        # measures["L1_path norm"] = calculate_path_norm(model, device, p=1, input_size=input_size) / margin**2

        # Bound Calculations
        alpha = math.sqrt(depth + math.log(nchannels * img_dim**2))
        bounds["L1_max Bound"] = alpha * measures["L_{1,inf} norm"] / math.sqrt(num_samples)
        bounds["Frobenius Bound"] = alpha * measures["Frobenius norm"] / math.sqrt(num_samples)

        # Enhanced Bounds
        beta = math.log(num_samples) * math.log(num_parameters)
        ratio_settings = {
            "model": model,
            "init_model": init_model,
            "measure_func": calculate_hidden_operator_norm,
            "operator": "norm",
        }
        ratio = calculate_measure(**ratio_settings, l=2 / 3, kwargs={"p": 2, "q": 1, "p_op": np.inf})
        bounds["Spec_L2_1 Bound"] = beta * spec_norm * ratio / math.sqrt(num_samples)
        ratio = calculate_measure(**ratio_settings, kwargs={"p": 2, "q": 2, "p_op": np.inf}, l=2)
        bounds["Spec_Fro Bound"] = beta * spec_norm * ratio / math.sqrt(num_parameters)

        # Flatness-Based Bounds
        sigma = calculate_path_norm(model, device, p=2, input_size=input_size)
        bounds["Flatness"] = 1 / (sigma**2)

    return {**measures, **bounds}
