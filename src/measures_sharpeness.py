from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from typing import Optional
import math
import numpy as np
from torch.utils.data import DataLoader
import time

# sources:
# https://github.com/facebookresearch/decodable_information_bottleneck
# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
# https://github.com/nitarshan/robust-generalization-measures
# Fantastic Generalization Measures and Where to Find Them: arXiv:1912.02178

# Automatically select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = torch.Generator(device=device)


@contextmanager
def apply_perturbation(
    model: nn.Module,
    sigma: float,
    noise_type: str = "gaussian",
    magnitude_eps: Optional[float] = None,
    uniform_range: Optional[float] = None,
):
    """
    Context manager to apply temporary perturbations (Gaussian or uniform) to a PyTorch model's parameters.
    """
    if noise_type not in ["gaussian", "uniform"]:
        raise ValueError("noise_type must be either 'gaussian' or 'uniform'")

    model = model.to(device)  # Ensure model is on the correct device
    original_state = deepcopy(model.state_dict())

    try:
        for name, param in model.named_parameters():
            if noise_type == "gaussian":
                if magnitude_eps is not None:
                    std = torch.sqrt(sigma**2 * torch.abs(param) ** 2 + magnitude_eps**2)
                    mean = torch.zeros_like(param)
                    noise = torch.normal(mean=mean, std=std, generator=rng)
                else:
                    std = sigma
                    noise = torch.normal(mean=0.0, std=std, size=param.size(), generator=rng, device=device)
            else:  # uniform
                if uniform_range is None:
                    raise ValueError("uniform_range must be specified for uniform noise")
                noise = torch.empty(param.size(), device=device).uniform_(-uniform_range, uniform_range, generator=rng)
            param.data.add_(noise)
        yield model
    finally:
        model.load_state_dict(original_state)


def calculate_perturbation_norm(model: nn.Module, original_state: dict) -> float:
    """
    Calculate the L2 norm of the perturbations applied to the model.
    """
    perturbations = []
    for name, param in model.named_parameters():
        original_param = original_state[name].to(device)
        perturbation = param.data - original_param
        perturbations.append(perturbation.view(-1))
    concatenated = torch.cat(perturbations)
    return torch.norm(concatenated).item()


def calculate_distance_vector(model: nn.Module, init_model: nn.Module) -> torch.Tensor:
    """
    Calculate the weight difference vector between the current and initial models.
    """
    dist_w_vec = []
    for param, init_param in zip(model.parameters(), init_model.parameters()):
        dist_w_vec.append((param.data - init_param.data).view(-1))
    return torch.cat(dist_w_vec)


def clamp_weights(
    model: nn.Module,
    original_weights: list,
    sigma: float,
    use_plus_one: bool = False,
):
    """
    Clamp model parameters to be within [original - (|original| + ac) * sigma, original + (|original| + ac) * sigma].
    """
    ac = 1.0 if use_plus_one else 0.0
    for original, param in zip(original_weights, model.parameters()):
        upper_bound = original + (torch.abs(original) + ac) * sigma
        lower_bound = original - (torch.abs(original) + ac) * sigma
        param.data.clamp_(lower_bound, upper_bound)


def normalize_parameter_perturbation(model, device, sigma, original_state, perturb_norm):
    for name, param in model.named_parameters():
        original_param = original_state[name].to(device)
        perturb = param.data - original_param
        perturb = perturb * (sigma / perturb_norm)
        param.data = original_param + perturb


def calculate_perturbed_accuracy(model, dataloader):
    model = model.to(device)  # Ensure model is on the correct device
    model.eval()
    batch_correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            batch_correct += pred.eq(target).sum().item()
    perturbed_accuracy = batch_correct / len(dataloader.dataset)
    return perturbed_accuracy


def gradient_ascent_step(model, dataloader, learning_rate):
    model = model.to(device)
    model.train()
    data_iter = iter(dataloader)
    data, target = next(data_iter)
    data, target = data.to(device), target.to(device)

    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()

    # Gradient ascent step (maximize loss)
    for param in model.parameters():
        if param.grad is not None:
            param.data += learning_rate * param.grad.data
    return model


def pac_bayes_sigma_search(
    model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    target_deviation: float,
    search_depth: int = 20,
    monte_carlo_iter: int = 15,
    magnitude_eps: float = None,
    upper: float = 2.0,
    lower: float = 0.0,
    deviation_tolerance: float = 1e-2,
    bound_tolerance: float = 1e-5,
):
    model = model.to(device)
    for _ in range(search_depth):
        sigma = (lower + upper) / 2
        accuracy_samples = []

        for _ in range(monte_carlo_iter):
            with apply_perturbation(model, sigma, magnitude_eps=magnitude_eps):
                perturbed_accuracy = calculate_perturbed_accuracy(model, dataloader)
                accuracy_samples.append(perturbed_accuracy)
        deviation = abs(np.mean(accuracy_samples) - accuracy)
        if abs(deviation - target_deviation) < deviation_tolerance or (upper - lower) < bound_tolerance:
            break
        elif deviation > target_deviation:
            upper = sigma
        else:
            lower = sigma

    return sigma


def sharpness_sigma_search(
    model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    target_deviation: float,
    learning_rate: float,
    ascent_steps: int = 20,
    device: str = device,
    magnitude_eps: float = None,
    noise_type: str = "gaussian",
    uniform_range: Optional[float] = None,
    search_depth: int = 20,
    monte_carlo_iter: int = 15,
    upper: float = 5.0,
    lower: float = 0.0,
    deviation_tolerance: float = 1e-2,
    bound_tolerance: float = 5e-3,
) -> float:
    model = model.to(device)
    for _ in range(search_depth):
        sigma = (lower + upper) / 2.0
        min_accuracy = float("inf")  # Initialize to infinity for min comparison

        for mt in range(monte_carlo_iter):
            with apply_perturbation(
                model,
                sigma,
                magnitude_eps=magnitude_eps,
                noise_type=noise_type,
                uniform_range=uniform_range,
            ):
                original_state = deepcopy(model.state_dict())
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                for _ in range(ascent_steps):
                    model = gradient_ascent_step(model, dataloader, learning_rate)
                    # Calculate perturbation norm
                    perturb_norm = calculate_perturbation_norm(model, original_state)
                    if perturb_norm > sigma:
                        normalize_parameter_perturbation(model, device, sigma, original_state, perturb_norm)

                model.eval()
                perturbed_accuracy = calculate_perturbed_accuracy(model, dataloader)
                min_accuracy = min(min_accuracy, perturbed_accuracy)

        deviation = abs(min_accuracy - accuracy)

        if abs(deviation - target_deviation) < deviation_tolerance or (upper - lower) < bound_tolerance:
            break
        elif deviation > target_deviation:
            upper = sigma
        else:
            lower = sigma

    return sigma


def magnitude_aware_sharpness_sigma(
    model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    target_deviation: float,
    learning_rate: float,
    ascent_steps: int = 20,
    device: str = device,
    search_depth: int = 20,
    monte_carlo_iter: int = 15,
    upper: float = 5.0,
    lower: float = 0.0,
    deviation_tolerance: float = 1e-2,
    bound_tolerance: float = 5e-3,
    use_plus_one: bool = False,
) -> float:
    model = model.to(device)
    original_weights = [param.data.clone() for param in model.parameters()]

    for j in range(search_depth):
        sigma = (upper + lower) / 2.0
        min_accuracy = float("inf")  # Initialize to infinity for min comparison

        for i in range(monte_carlo_iter):
            with apply_perturbation(
                model=model,
                sigma=sigma,
                noise_type="uniform",
                uniform_range=sigma,
            ):
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                clamp_weights(model, original_weights, sigma, use_plus_one)

                for _ in range(ascent_steps):
                    model = gradient_ascent_step(model, dataloader, learning_rate)
                    clamp_weights(model, original_weights, sigma, use_plus_one)

            perturbed_accuracy = calculate_perturbed_accuracy(model, dataloader)
            min_accuracy = min(min_accuracy, perturbed_accuracy)

        deviation = abs(min_accuracy - accuracy)

        if abs(deviation - target_deviation) < deviation_tolerance or (upper - lower) < bound_tolerance:
            break
        elif deviation > target_deviation:
            upper = sigma
        else:
            lower = sigma

    return sigma


def calculate_pac_bayes_metrics(model, init_model, dataloader, accuracy: float):
    model = model.to(device)
    init_model = init_model.to(device)

    dataset_size = len(dataloader.dataset)
    measures = {}
    num_params = sum(p.numel() for p in model.parameters())

    # PAC-Bayes Sigma
    sigma_search_settings = {
        "model": model,
        "dataloader": dataloader,
        "accuracy": accuracy,
        "target_deviation": 0.01,
    }

    mag_eps = 1e-3
    sigma = pac_bayes_sigma_search(**sigma_search_settings)
    mag_sigma = pac_bayes_sigma_search(**sigma_search_settings, magnitude_eps=mag_eps)

    distance_vector = calculate_distance_vector(model, init_model)

    def _pacbayes_bound(reference_vec: torch.Tensor, sigma: float) -> float:
        return (reference_vec.norm(p=2) ** 2) / (4 * sigma**2) + math.log(dataset_size / sigma) + 10

    def _pacbayes_mag_bound(reference_vec: torch.Tensor, sigma: float) -> float:
        numerator = mag_eps**2 + (sigma**2 + 1) * (reference_vec.norm(p=2) ** 2) / num_params
        denominator = mag_eps**2 + sigma**2 * distance_vector.norm(p=2) ** 2
        return 0.25 * (numerator / denominator).log() + math.log(dataset_size / sigma) + 10

    measures["PAC_Bayes_Sigma"] = sigma
    measures["PAC_Bayes_Bound"] = float(_pacbayes_bound(distance_vector, sigma))
    measures["PAC_Bayes_Flatness"] = 1 / sigma**2

    measures["PAC_Bayes_MAG_Sigma"] = mag_sigma
    measures["PAC_Bayes_MAG_Bound"] = float(_pacbayes_mag_bound(distance_vector, mag_sigma))
    measures["PAC_Bayes_MAG_Flatness"] = 1 / mag_sigma**2

    return measures


def calculate_sharpness_metrics(model, dataloader, accuracy: float):
    model = model.to(device)
    measures = {}

    num_params = sum(p.numel() for p in model.parameters())

    # Sharpness Sigma
    sigma_search_settings = {
        "model": model,
        "dataloader": dataloader,
        "accuracy": accuracy,
        "target_deviation": 0.01,
        "learning_rate": 0.01,  # You may need to adjust these hyperparams
    }

    mag_eps = 1e-3
    sharpness_sigma = sharpness_sigma_search(**sigma_search_settings)
    sharpness_mag_sigma = sharpness_sigma_search(**sigma_search_settings, magnitude_eps=mag_eps)

    def sharpness_bound(sigma: float) -> float:
        return math.log(num_params) * math.sqrt(1 / sigma**2)

    measures["Sharpness_Sigma"] = sharpness_sigma
    measures["Sharpness_Flatness"] = 1 / sharpness_sigma**2
    measures["Sharpness_Bound"] = sharpness_bound(sharpness_sigma)

    measures["Sharpness_MAG_Sigma"] = sharpness_mag_sigma
    measures["Sharpness_MAG_Flatness"] = 1 / sharpness_mag_sigma**2
    measures["Sharpness_MAG_Bound"] = sharpness_bound(sharpness_mag_sigma)

    return measures


def calculate_combined_metrics(model, init_model, dataloader, accuracy: float, num_params: int):
    model = model.to(device)
    init_model = init_model.to(device)

    pac_bayes_metrics = calculate_pac_bayes_metrics(model, init_model, dataloader, accuracy)
    sharpness_metrics = calculate_sharpness_metrics(model, dataloader, accuracy)

    return {**pac_bayes_metrics, **sharpness_metrics}
