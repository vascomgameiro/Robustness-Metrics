import torch
import os
import copy
from torch import nn
from pytorch_trainer import PyTorchTrainer
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path


def train_save_model(config, device, train_loader, val_loader):
    model = config["model"]
    model_name = config["name"]
    optimizer = config["params"]["optimizer"]
    lr = config["params"]["lr"]

    untrained = copy.deepcopy(model)
    # the current working dir should be the project root: robustness-metrics
    path_to_model = f"models/{model_name}"
    path_to_plots = os.path.join(path_to_model, "plots")

    if not os.path.exists(f"{path_to_model}/trained.pt"):
        optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        optim_cls = optims[optimizer]
        if lr == "scheduler":
            lr = 0.01
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_cls(model.parameters(), lr=lr), mode="min", factor=0.5, patience=2
            )
        else:
            scheduler = None
        os.makedirs(path_to_model, exist_ok=True)
        trainer = PyTorchTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim_cls(model.parameters(), lr=lr),
            scheduler=scheduler,
            device=device,
        )

        torch.save(model.state_dict(), f"{path_to_model}/untrained.pt")

        trainer.train(num_epochs=100, early_stopping_patience=10)
        model = trainer.best_model
        print(next(model.parameters()).device)
        trainer.save_best_model(path_to_model)
        trainer.save_plots(path_to_plots)

    else:
        print(f"Model {model_name} is already trained. Skipping training step.")

        model.load_state_dict(torch.load(f"{path_to_model}/trained.pt", map_location=device))
        untrained.load_state_dict(torch.load(f"{path_to_model}/untrained.pt", map_location=device))

        model = model.to(device)
        untrained = untrained.to(device)

    return untrained, model


def get_logits_and_labels(
    dataloader: DataLoader,
    model: torch.nn.Module,
    model_name: str,
    set_name: str,
    device: torch.device,
    save_dir: str = "models",
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Compute logits and labels for a dataloader and save logits.

    Args:
        dataloader (DataLoader): Dataset loader.
        model (torch.nn.Module): Trained model.
        model_name (str): Model name for saving.
        set_name (str): Dataset name (e.g., train/test).
        device (torch.device): Computation device.
        save_dir (str): Directory to save logits.

    Returns:
        tuple[np.ndarray, torch.Tensor]: Logits and labels.
    """
    save_path = Path(save_dir) / model_name / f"predictions/predictions_{set_name}.npy"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(save_path):
        model.to(device).eval()
        logits, labels = [], []

        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.to(device), label.to(device)
                logits.append(model(data).cpu())
                labels.append(label.cpu())

        logits = torch.cat(logits).numpy()
        labels = torch.cat(labels).numpy()

        np.save(save_path, logits)
    else:
        logits = np.load(save_path)
        labels = []

        for _, label in dataloader:
            labels.append(label.cpu())
        labels = torch.cat(labels).numpy()

    return logits, labels


def load_logits_and_labels_attacks(
    model_name: str, set_name: str, save_dir: str = "attacks"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load logits and labels as NumPy arrays from a saved file.

    Args:
        model_name (str): Model name for locating the saved file.
        set_name (str): Dataset name (e.g., train/test).
        save_dir (str): Directory where the logits and labels are stored.

    Returns:
        tuple[np.ndarray, np.ndarray]: Logits and labels as NumPy arrays.
    """
    save_path = Path(save_dir) / model_name / f"predictions_{set_name}.pt"
    if not save_path.exists():
        raise FileNotFoundError(f"File not found: {save_path}")

    data = torch.load(save_path)
    logits = data["logits"].numpy()
    labels = data["labels"].numpy()
    return logits, labels


def calculate_metric_differences(complex_cifar, complex_val):
    """
    Calculate absolute difference, proportional difference, and elasticity for each metric.

    Args:
        complex_cifar (dict): Dictionary containing training metrics.
        complex_val (dict): Dictionary containing validation metrics.

    Returns:
        dict: Dictionary containing the calculated differences for each metric.
    """
    differences = {}

    for metric_name in complex_cifar:
        train_metric = complex_cifar[metric_name]
        val_metric = complex_val[metric_name]

        abs_diff = abs(train_metric - val_metric)
        prop_diff = abs(train_metric - val_metric) / train_metric if train_metric != 0 else float("inf")
        elasticity = abs(train_metric - val_metric) / ((val_metric + train_metric) / 2)

        differences[metric_name] = {
            "absolute_difference": abs_diff,
            "proportional_difference": prop_diff,
            "elasticity": elasticity,
        }

    return differences


def print_save_measures(dic, statement, path_save):
    print("\n")
    print(statement)
    for key, value in dic.items():
        print(f"{key}: {value}")
    print(f"Dictionary saved to {path_save}")
    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    torch.save(dic, path_save)
