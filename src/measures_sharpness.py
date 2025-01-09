from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from typing import Optional
import math
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Literal
import random

# sources:
# https://github.com/facebookresearch/decodable_information_bottleneck
# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
# https://github.com/nitarshan/robust-generalization-measures
# Fantastic Generalization Measures and Where to Find Them: arXiv:1912.02178

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = torch.Generator(device=device.type)


@contextmanager
def apply_perturbation(
    model: nn.Module,
    sigma: float,
    noise_type: Literal["gaussian_standard", "gaussian_magnitude_aware", "uniform_standard", "uniform_magnitude_aware"],
    magnitude_eps: float = 1e-3,
):
    """
    Context manager to apply temporary perturbations to a PyTorch model's parameters.
    Supports:
      - "gaussian_standard": noise ~ N(0, sigma^2).
      - "gaussian_magnitude_aware": noise ~ N(0, sigma^2 * |param|^2 + magnitude_eps^2).
      - "uniform_standard": noise ~ U(-sigma/2, sigma/2).
      - "uniform_magnitude_aware": noise ~ U(-sigma/2  * |param|, sigma/2  * |param|).
    """
    valid_noise_types = [
        "gaussian_standard",
        "gaussian_magnitude_aware",
        "uniform_standard",
        "uniform_magnitude_aware",
    ]
    if noise_type not in valid_noise_types:
        raise ValueError(f"noise_type must be one of {valid_noise_types}, got '{noise_type}'")

    model = model.to(device)  # Ensure model is on the correct device
    original_state = deepcopy(model.state_dict())

    try:
        for name, param in model.named_parameters():
            if noise_type == "gaussian_magnitude_aware":
                std = torch.sqrt(sigma**2 * torch.abs(param) ** 2 + magnitude_eps**2)
                mean = torch.zeros_like(param)
                noise = torch.normal(mean=mean, std=std, generator=rng)

            elif noise_type == "gaussian_standard":
                std = sigma
                noise = torch.normal(mean=0.0, std=std, size=param.size(), generator=rng, device=device)

            elif noise_type == "uniform_magnitude_aware":
                unif = torch.distributions.uniform.Uniform(-param.abs() * sigma, param.abs() * sigma)
                noise = unif.sample()

            else:  # "uniform_standard"
                noise = torch.empty(param.size(), device=device).uniform_(-sigma, sigma, generator=rng)

            param.data.add_(noise)
        yield model
    finally:
        model.load_state_dict(original_state)


def calculate_distance_vector(model: nn.Module, init_model: nn.Module) -> torch.Tensor:
    """
    Calculate the L2 distance vector between current and initial model parameters.
    """
    distance = []
    for name, param in model.state_dict().items():
        distance.append((param - init_model.state_dict()[name]).view(-1))
    return torch.cat(distance)


def calculate_perturbed_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Compute accuracy of the perturbed model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total


def clip_perturbed_weights(
    model: nn.Module,
    sigma: float,
    original_weights: list,
):
    """
    Clamp model parameters within [original - (|original|) * sigma, original + (|original|) * sigma].

    Parameters:
    - model: PyTorch model.
    - sigma: Perturbation scale.
    - original_weights: List of original parameter tensors.
    """

    for original, param in zip(original_weights, model.parameters()):
        upper = original + (torch.abs(original)) * sigma
        lower = original - (torch.abs(original)) * sigma
        param.data.clamp_(lower, upper)


def normalize_perturbed_weights(
    perturbed_model: nn.Module,
    unperturbed_model: nn.Module,
    sigma: float,
):
    """
    Scale perturbations to ensure total L2 norm does not exceed sigma.

    Parameters:
    - perturbed_model: Model with perturbed weights.
    - unperturbed_model: Model with original weights.
    - sigma: Perturbation scale.
    """

    # Ensure both models are on the specified device
    unperturbed_model = unperturbed_model.to(device)
    perturbed_model = perturbed_model.to(device)

    total_squared_diff = 0.0
    with torch.no_grad():
        for param_orig, param_pert in zip(unperturbed_model.parameters(), perturbed_model.parameters()):
            diff = param_pert - param_orig
            total_squared_diff += torch.sum(diff**2).item()

    l2_norm = total_squared_diff**0.5
    print(f"Weights Norm before normalzing {l2_norm}")
    if l2_norm > sigma:
        scaling_factor = sigma / l2_norm
        with torch.no_grad():
            for param_orig, param_pert in zip(unperturbed_model.parameters(), perturbed_model.parameters()):
                scaled_diff = (param_pert - param_orig) * scaling_factor
                param_pert.copy_(param_orig + scaled_diff)
        # Verify norm after normalizing
        total_squared_diff = 0.0
        with torch.no_grad():
            for param_orig, param_pert in zip(unperturbed_model.parameters(), perturbed_model.parameters()):
                diff = param_pert - param_orig
                total_squared_diff += torch.sum(diff**2).item()

        print(f"Weights Norm after normalzing {total_squared_diff**0.5}")


def gradient_ascent_one_step(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Perform one gradient ascent step using multiple randomly selected batches.

    Parameters:
    - model: The PyTorch model to be updated.
    - dataloader: DataLoader providing the dataset.
    - optimizer: Optimizer for updating the model parameters.
    - device: The device (CPU/GPU) to perform computations on.
    - num_batches: Number of batches to process in this ascent step.

    Returns:
    - average_loss: The average loss over the selected batches.
    """
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)

        # Gradient ascent
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()

    return model


def sharpness_sigma_search(
    model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    target_deviation: float,
    device: torch.device,
    noise_type: Literal["uniform_standard", "uniform_magnitude_aware"],
    search_depth: int = 50,
    monte_carlo_iter: int = 5,
    upper: float = 5.0,
    lower: float = 0.0,
    deviation_tolerance: float = 1e-2,
    bound_tolerance: float = 1e-2,
    learning_rate: float = 0.01,
    ascent_steps: int = 5,
) -> float:
    model.to(device)

    original_weights = [param.detach().clone() for param in model.parameters()]
    print(f"Accuracy: {accuracy}")
    for sd in range(search_depth):
        sigma = (lower + upper) / 2.0
        min_accuracy = float("inf")
        early_stoping = False
        print(f"Search depth {sd}, sigma: {sigma}")

        for mt in range(monte_carlo_iter):
            print(f"Monte Carlo iteration {mt}")
            # original_model = deepcopy(model)

            with apply_perturbation(model, sigma, noise_type=noise_type):
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                for step in range(ascent_steps):
                    print(f"Step: {step}")
                    gradient_ascent_one_step(model, dataloader, optimizer, device)
                    clip_perturbed_weights(model, sigma, original_weights)
                    # if noise_type == "uniform_magnitude_aware":
                    #     clip_perturbed_weights(model, sigma, original_weights)
                    # else:  # uniform standard
                    #     normalize_perturbed_weights(model, original_model, sigma)

                    # Check if deviation is too far from target; trigger early stopping
                    if (step + 1) % 5 == 0 or step == 1 or step == 0:
                        perturbed_accuracy = calculate_perturbed_accuracy(model, dataloader, device)
                        current_deviation = accuracy - perturbed_accuracy
                        print(f"Deviation: {current_deviation}, Perturbed Accuracy: {perturbed_accuracy}")

                        if current_deviation > target_deviation + deviation_tolerance and step != 0:
                            early_stoping = True
                            print("Early stopping triggered.")
                            break

                min_accuracy = min(min_accuracy, perturbed_accuracy)
            if early_stoping:
                break

        deviation = abs(min_accuracy - accuracy)
        print(f"Min accuracy: {min_accuracy}, Deviation: {deviation}")
        # Check if within tolerance
        if abs(deviation - target_deviation) < deviation_tolerance or (upper - lower) < bound_tolerance:
            print("Desired deviation achieved or search bounds converged.")
            break

        # Dynamic sigma adjustment
        if deviation > target_deviation:
            upper = sigma
        else:
            lower = sigma
        print(f"Lower bound: {lower}, Upper bound: {upper}")
        print("\n")

    print(f"Final sigma: {sigma}")
    return sigma


def pac_bayes_sigma_search(
    model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    device: str,
    noise_type: Literal["gaussian_standard", "gaussian_magnitude_aware"],
    target_deviation: float = 0.1,
    search_depth: int = 15,
    monte_carlo_iter: int = 15,
    upper: float = 5.0,
    lower: float = 0.0,
    deviation_tolerance: float = 1e-3,
    bound_tolerance: float = 1e-3,
):
    model = model.to(device)
    for _ in range(search_depth):
        sigma = (lower + upper) / 2
        accuracy_samples = []
        for _ in range(monte_carlo_iter):
            with apply_perturbation(model, sigma, noise_type):
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


def pacbayes_bound(reference_vec: torch.Tensor, sigma: float, dataset_size: int) -> float:
    return (reference_vec.norm(p=2) ** 2) / (4 * sigma**2) + math.log(dataset_size / sigma) + 10


def pacbayes_mag_bound(
    reference_vec: torch.Tensor,
    sigma: float,
    distance_vector: torch.Tensor,
    mag_eps: float,
    num_params: int,
    dataset_size: int,
) -> float:
    numerator = mag_eps**2 + (sigma**2 + 1) * (reference_vec.norm(p=2) ** 2) / num_params
    denominator = mag_eps**2 + sigma**2 * distance_vector.norm(p=2) ** 2
    return 0.25 * torch.log(numerator / denominator).item() + math.log(dataset_size / sigma) + 10


def calculate_pac_bayes_metrics(
    model: nn.Module,
    init_model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    mag_eps: float = 1e-3,
    device: torch.device = device,
) -> dict:
    """
    Calculate PAC-Bayes metrics.
    """
    model.to(device)
    init_model.to(device)

    dataset_size = len(dataloader.dataset)
    num_params = sum(p.numel() for p in model.parameters())
    measures = {}

    sigma = pac_bayes_sigma_search(
        model=model,
        dataloader=dataloader,
        accuracy=accuracy,
        target_deviation=0.1,
        device=device,
        noise_type="gaussian_standard",
        upper=5.0,
        lower=0.0,
        deviation_tolerance=1e-2,
        bound_tolerance=1e-3,
        search_depth=50,
    )
    mag_sigma = pac_bayes_sigma_search(
        model=model,
        dataloader=dataloader,
        accuracy=accuracy,
        target_deviation=0.1,
        device=device,
        noise_type="gaussian_magnitude_aware",
        upper=1.0,
        lower=0.0,
        deviation_tolerance=1e-2,
        bound_tolerance=1e-5,
        search_depth=50,
    )

    distance_vector = calculate_distance_vector(model, init_model)

    measures["PAC_Bayes_Sigma"] = sigma
    measures["PAC_Bayes_Bound"] = pacbayes_bound(distance_vector, sigma, dataset_size)
    measures["PAC_Bayes_Flatness"] = 1 / sigma**2

    measures["PAC_Bayes_MAG_Sigma"] = mag_sigma
    measures["PAC_Bayes_MAG_Bound"] = pacbayes_mag_bound(
        distance_vector, mag_sigma, distance_vector, mag_eps, num_params
    )
    measures["PAC_Bayes_MAG_Flatness"] = 1 / mag_sigma**2

    return measures


def sharpness_bound(sigma: float, num_params: int) -> float:
    return math.log(num_params) * math.sqrt(1 / sigma**2)


def calculate_sharpness_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    device: torch.device = device,
    lr: float = 0.001,
) -> dict:
    """
    Calculate sharpness metrics.
    """
    model.to(device)
    measures = {}
    num_params = sum(p.numel() for p in model.parameters())

    sharpness_sigma = sharpness_sigma_search(
        model=model,
        dataloader=dataloader,
        accuracy=accuracy,
        target_deviation=0.1,
        device=device,
        noise_type="uniform_standard",
        upper=1.0,
        lower=0.0,
        deviation_tolerance=1e-2,
        bound_tolerance=1e-5,
        learning_rate=lr,
        ascent_steps=20,
        monte_carlo_iter=5,
        search_depth=50,
    )

    sharpness_mag_sigma = sharpness_sigma_search(
        model=model,
        dataloader=dataloader,
        accuracy=accuracy,
        target_deviation=0.1,
        device=device,
        noise_type="uniform_magnitude_aware",
        upper=1.0,
        lower=0.0,
        deviation_tolerance=1e-2,
        bound_tolerance=1e-5,
        learning_rate=lr,
        ascent_steps=20,
        monte_carlo_iter=5,
        search_depth=50,
    )

    measures["Sharpness_Sigma"] = sharpness_sigma
    measures["Sharpness_Flatness"] = 1 / sharpness_sigma**2
    measures["Sharpness_Bound"] = sharpness_bound(sharpness_sigma, num_params)

    measures["Sharpness_MAG_Sigma"] = sharpness_mag_sigma
    measures["Sharpness_MAG_Flatness"] = 1 / sharpness_mag_sigma**2
    measures["Sharpness_MAG_Bound"] = sharpness_bound(sharpness_mag_sigma, num_params)

    return measures


def calculate_combined_metrics(
    model: nn.Module,
    init_model: nn.Module,
    dataloader: DataLoader,
    accuracy: float,
    mag_eps: float = 1e-3,
    lr: float = 0.001,
    device: torch.device = device,
) -> dict:
    """
    Combine PAC-Bayes and Sharpness metrics.
    """
    model.to(device)
    init_model.to(device)

    pac_bayes_metrics = calculate_pac_bayes_metrics(model, init_model, dataloader, accuracy, mag_eps, device)
    sharpness_metrics = calculate_sharpness_metrics(model, dataloader, accuracy, device, lr)

    return {**pac_bayes_metrics, **sharpness_metrics}
