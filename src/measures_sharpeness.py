from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from typing import Optional
import math
import numpy as np


@contextmanager
def apply_perturbation(model: nn.Module, sigma: float, rng: torch.Generator, magnitude_eps: Optional[float] = None):
    """
    Context manager to apply temporary Gaussian perturbations to a PyTorch model's parameters.

    Args:
        model (nn.Module): The PyTorch model to perturb.
        sigma (float): Standard deviation of the Gaussian noise.
        rng (torch.Generator): Random number generator for reproducibility.
        magnitude_eps (Optional[float]): Small constant for scaling the noise (default: None).

    Yields:
        nn.Module: A perturbed copy of the model.

    Ensures:
        The original model remains unmodified after exiting the context.
    """
    # Ensure noise is applied on the same device as model parameters
    device = next(model.parameters()).device

    # Generate noise for each parameter
    noise = []
    for param in model.parameters():
        if magnitude_eps is not None:
            std = (sigma**2 * torch.abs(param) ** 2 + magnitude_eps**2).sqrt()
            noise.append(torch.normal(mean=0, std=std, generator=rng).to(device))
        else:
            noise.append(torch.normal(mean=0, std=sigma, size=param.size(), generator=rng, device=device))

    # Apply perturbations
    perturbed_model = deepcopy(model)
    try:
        for param, n in zip(perturbed_model.parameters(), noise):
            param.data.add_(n)
        yield perturbed_model
    finally:
        del perturbed_model


def calculate_dist_w_vec(model, init_model):
    """
    Calculate the distance vector between the current model's weights and the initialized model's weights.

    Args:
        model (torch.nn.Module): The trained model.
        init_model (torch.nn.Module): The initialized model.

    Returns:
        torch.Tensor: Flattened distance vector of weights.
    """
    dist_w_vec = []
    for param, init_param in zip(model.parameters(), init_model.parameters()):
        dist_w_vec.append((param.data - init_param.data).view(-1))
    return torch.cat(dist_w_vec)


def pac_bayes_sigma_search(model, dataloader, accuracy: float, target_deviation: float, magnitude_eps: float = None):
    """
    Binary search for sigma based on accuracy deviation.

    Args:
        accuracy (float): Target model accuracy.
        target_deviation (float): Target deviation for perturbed accuracy.
        magnitude_eps (float): Magnitude scaling factor for noise.

    Returns:
        float: Optimal sigma value.
    """
    lower, upper = 0.0, 2.0
    tolerance = 1e-2
    sigma = 1.0

    for _ in range(20):  # Search depth
        sigma = (lower + upper) / 2
        accuracy_samples = []

        for _ in range(15):  # Monte Carlo iterations
            with apply_perturbation(model, sigma, magnitude_eps):
                batch_correct = 0
                for data, target in dataloader:
                    output = model(data)
                    pred = output.argmax(dim=1)
                    batch_correct += pred.eq(target).sum().item()
                perturbed_accuracy = batch_correct / len(dataloader.dataset)
                accuracy_samples.append(perturbed_accuracy)

        deviation = abs(np.mean(accuracy_samples) - accuracy)
        if abs(deviation - target_deviation) < tolerance:
            break
        elif deviation > target_deviation:
            upper = sigma
        else:
            lower = sigma

    return sigma


def calculate_pac_bayes_metrics(model, init_model, dataloader, accuracy: float, num_params: int):
    """
    Calculate PAC-Bayes generalization bounds, flatness metrics, and sharpness measures.

    Args:
        acc (float): Model accuracy.
        num_params (int): Number of model parameters.
        dist_w_vec (torch.Tensor): Weight vector difference from initialization.
        w_vec (torch.Tensor): Current weight vector.

    Returns:
        Dict[str, Any]: Dictionary of calculated metrics.
    """
    dataset_size = len(dataloader.dataset)
    measures = {}

    # PAC-Bayes Sigma
    sigma_search_settings = {
        "model": model,
        "dataloader": dataloader,
        "accuracy": accuracy,
        "target_deviation": 0.01,
    }

    mag_eps = 1e-3
    pac_sigma = pac_bayes_sigma_search(**sigma_search_settings)
    mag_sigma = pac_bayes_sigma_search(**sigma_search_settings, magnitude_eps=mag_eps)

    dist_w_vec = calculate_dist_w_vec(model, init_model)

    # PAC-Bayes Bound
    def _pacbayes_bound(reference_vec: torch.Tensor, sigma: float) -> float:
        return (reference_vec.norm(p=2) ** 2) / (4 * sigma**2) + math.log(dataset_size / sigma) + 10

    # Magnitude-Aware PAC-Bayes Bound
    def _pacbayes_mag_bound(reference_vec: torch.Tensor, sigma: float) -> float:
        numerator = mag_eps**2 + (sigma**2 + 1) * (reference_vec.norm(p=2) ** 2) / num_params
        denominator = mag_eps**2 + sigma**2 * dist_w_vec.norm(p=2) ** 2
        return 0.25 * (numerator / denominator).log() + math.log(dataset_size / sigma) + 10

    # Metrics
    measures["PAC_Bayes_Sigma"] = pac_sigma
    measures["PAC_Bayes_Bound"] = _pacbayes_bound(dist_w_vec, pac_sigma)
    measures["PAC_Bayes_Flatness"] = 1 / pac_sigma**2
    measures["PAC_Bayes_MAG_Sigma"] = mag_sigma
    measures["PAC_Bayes_MAG_Bound"] = _pacbayes_mag_bound(dist_w_vec, mag_sigma)
    measures["PAC_Bayes_MAG_Flatness"] = 1 / mag_sigma**2

    return measures


def sharpness_sigma_search(model, dataloader, accuracy: float, target_deviation: float, magnitude_eps: float = None):
    """
    Binary search for sharpness sigma based on accuracy deviation.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for the dataset.
        accuracy (float): Target model accuracy.
        target_deviation (float): Target deviation for perturbed accuracy.
        ascent_steps (int): Number of ascent steps for perturbation optimization.

    Returns:
        float: Optimal sharpness sigma.
    """
    lower, upper = 0.0, 5.0
    tolerance = 1e-3
    sigma = 2.5

    for _ in range(20):  # Search depth
        sigma = (lower + upper) / 2
        min_accuracy = 10.0

        for _ in range(15):  # Monte Carlo iterations
            with apply_perturbation(model, sigma, 42, magnitude_eps):
                batch_correct = 0
                for data, target in dataloader:
                    data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    batch_correct += pred.eq(target).sum().item()
                perturbed_accuracy = batch_correct / len(dataloader.dataset)

                # Update minimum accuracy deviation
                min_accuracy = min(min_accuracy, perturbed_accuracy)

        deviation = abs(min_accuracy - accuracy)
        if abs(deviation - target_deviation) < tolerance:
            break
        elif deviation > target_deviation:
            upper = sigma
        else:
            lower = sigma

    return sigma


def calculate_sharpness_metrics(model, dataloader, accuracy: float, num_params: int):
    """
    Calculate sharpness metrics based on sharpness sigma.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): DataLoader for the dataset.
        accuracy (float): Model accuracy.
        num_params (int): Number of model parameters.

    Returns:
        Dict[str, Any]: Dictionary of sharpness metrics.
    """
    measures = {}

    # Sharpness Sigma
    sharpness_sigma = sharpness_sigma_search(model, dataloader, accuracy, target_deviation=0.01)

    sigma_search_settings = {
        "model": model,
        "dataloader": dataloader,
        "accuracy": accuracy,
        "target_deviation": 0.01,
    }

    mag_eps = 1e-3
    sharpness_sigma = sharpness_sigma_search(**sigma_search_settings)
    sharpness_mag_sigma = sharpness_sigma_search(**sigma_search_settings, magnitude_eps=mag_eps)

    # Sharpness Bound
    def sharpness_bound(sigma: float) -> float:
        return math.log(num_params) * math.sqrt(1 / sigma**2)

    # Metrics
    measures["Sharpness_Sigma"] = sharpness_sigma
    measures["Sharpness_Flatness"] = 1 / sharpness_sigma**2
    measures["Sharpness_Bound"] = sharpness_bound(sharpness_sigma)
    measures["Sharpness_MAG_Sigma"] = sharpness_mag_sigma
    measures["Sharpness_MAG_Flatness"] = 1 / sharpness_mag_sigma**2
    measures["Sharpness_MAG_Bound"] = sharpness_bound(sharpness_mag_sigma)

    return measures


# Combine PAC-Bayes and Sharpness Metrics
def calculate_combined_metrics(model, init_model, dataloader, accuracy: float, num_params: int):
    """
    Calculate combined PAC-Bayes and Sharpness metrics.

    Args:
        model (nn.Module): The PyTorch model.
        init_model (nn.Module): The initialized PyTorch model.
        dataloader (DataLoader): DataLoader for the dataset.
        accuracy (float): Model accuracy.
        num_params (int): Number of model parameters.

    Returns:
        Dict[str, Any]: Combined dictionary of PAC-Bayes and sharpness metrics.
    """
    pac_bayes_metrics = calculate_pac_bayes_metrics(model, init_model, dataloader, accuracy, num_params)
    sharpness_metrics = calculate_sharpness_metrics(model, dataloader, accuracy, num_params)

    return {**pac_bayes_metrics, **sharpness_metrics}
