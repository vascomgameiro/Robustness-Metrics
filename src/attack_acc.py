import torch
import torchvision.models as models
from torch import nn, optim

from model_mod import modify_last_layer, SimpleCNN
from data_loader import dataloader
from pytorch_trainer import PyTorchTrainer
from adversarial_attacks import (
    fgsm_attack,
    pgd_attack,
    cw_attack,
    bim_attack,
    square_attack,
    deepfool_attack,
    evaluate_attack,
)
import os

train_tiny_path = "/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/data/processed/train_tiny.pt"
val_tiny_path = "/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/data/processed/val_tiny.pt"
test_tiny_path = "/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/data/processed/test_tiny.pt"
r_path = "/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/data/processed/test_R.pt"

#this really is not needed
#train_loader, val_loader, test_loader_tiny, test_loader_r = dataloader(
#    train_tiny_path, val_tiny_path, test_tiny_path, r_path
#)

models_to_train = [
    {"name": "simple_cnn", "model": SimpleCNN, "params": {"lr": 0.001}}
    # {"name": "mobilenet_v3", "model": models.mobilenet_v3_small, "params": {"lr": 0.001}}  # ,
    # {"name": "resnet18", "model": models.resnet18, "params": {"lr": 0.0001}},
]


attacks_to_test = [
    {"name": "FGSM attack", "model": fgsm_attack, "params": {"eps": 0.1}},
    {"name": "PGD attack", "model": pgd_attack, "params": {"eps": 0.1, "alpha": 0.01, "steps": 5}},
    # {'name': 'CW attack', 'model': cw_attack, 'params': {'confidence': 10, 'steps': 3, 'lr': 0.1}},
    {"name": "DeepFool attack", "model": deepfool_attack, "params": {"overshoot": 0.02, "max_iter": 3}},
    # {'name': 'BIM attack', 'model': bim_attack, 'params': {'eps': 0.1, 'alpha': 0.01, 'steps': 3}},
    # {'name': 'Square Attack', 'model': square_attack, 'params': {'max_queries': 10000, 'eps': 0.1}},
]


# save_dir='/models/Attacked_tensors/'


import torch


def generate_and_save_attack(model, test_loader, attack_fn, attack_name, params, model_name, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_adv_images = []
    all_labels = []
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating adversarial examples for {model_name} using {attack_name}...")

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, **params)

        all_adv_images.append(adv_images.detach().cpu())
        all_labels.append(labels.detach().cpu())  # labels stay the same

    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save the adversarial examples and labels
    file_name = f"{model_name}_{attack_name}_{'_'.join(f'{k}_{v}' for k, v in params.items())}.pt"
    save_path = os.path.join(save_dir, file_name)
    torch.save({"adv_images": all_adv_images, "labels": all_labels}, save_path)
    print(f"Saved {attack_name} adversarial images for {model_name} to {save_path}.")


def evaluate_and_save_all_attacks(
    models_to_train,
    test_loader_tiny,
    attacks_to_test,
    save_dir="/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/Attacked_tensors/",
):
    """
    Evaluate all attacks on all models, save adversarial examples, and calculate adversarial accuracy.
    """

    for model_config in models_to_train:
        model_name = model_config["name"]
        model = model_config["model"]
        #model_params = model_config["params"]

        # Instantiate the model
        #model = model_class()

        # Train or load pre-trained weights (adjust path as necessary)
        model_path = f"models/{model_name}/{model_name}.pt"
        print(model_path)
        try:
            model.load_state_dict(torch.load(model_path))  # Load pre-trained weights
            print(f"Loaded pre-trained weights for {model_name} from {model_path}.")
        except FileNotFoundError:
            print(f"No pre-trained weights found for {model_name}, skipping.")
            continue

        # Evaluate all attacks for this model
        print(f"Evaluating attacks on {model_name}...")

        attacks_dir = os.path.join(save_dir, f"{model_name}/attacks")

        for config in attacks_to_test:
            attack_name = config["name"]
            attack_fn = config["model"]
            params = config["params"]

            # Generate and Save Adversarial Examples
            generate_and_save_attack(model, test_loader_tiny, attack_fn, attack_name, params, model_name, attacks_dir)

            # Load Saved Adversarial
            adv_images_path = (
                f"{attacks_dir}/{model_name}_{attack_name}_{'_'.join(f'{k}_{v}' for k, v in params.items())}.pt"
            )
            print(adv_images_path)
            if os.path.exists(adv_images_path):
                saved_data = torch.load(adv_images_path)  # Load the saved .pt file
                adv_images = saved_data["adv_images"]  # Extract adversarial images
                labels = saved_data["labels"]  # Extract labels
                print(f"Loaded adversarial examples for {attack_name} attack on {model_name} from {adv_images_path}.")

                adversarial_accuracy = evaluate_attack(model, adv_images, labels)  # Pass tensors directly
                print(
                    f"Accuracy on the test set after {attack_name} attack for {model_name}: {adversarial_accuracy:.2f}%"
                )
            else:
                print(
                    f"Adversarial examples for {attack_name} attack on {model_name} not found. Skipping accuracy calculation."
                )
                continue


def evaluate_attacks_saved(
    models_to_train,
    test_loader_tiny,
    attacks_to_test,
    save_dir="/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/Attacked_tensors/",
):
    """
    Evaluate all attacks on all models, save adversarial examples, and calculate adversarial accuracy.
    """

    for model_config in models_to_train:
        model_name = model_config["name"]
        model_class = model_config["model"]
        model_params = model_config["params"]

        model = model_class()

        model_path = f"/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/{model_name}.pt"
        print(model_path)
        try:
            model.load_state_dict(torch.load(model_path))  # Load pre-trained weights
            print(f"Loaded pre-trained weights for {model_name} from {model_path}.")
        except FileNotFoundError:
            print(f"No pre-trained weights found for {model_name}, skipping.")
            continue

        print(f"Evaluating attacks on {model_name}...")
        for config in attacks_to_test:
            attack_name = config["name"]
            attack_fn = config["model"]
            params = config["params"]

            adv_images_path = (
                f"{save_dir}{model_name}_{attack_name}_{'_'.join(f'{k}_{v}' for k, v in params.items())}.pt"
            )
            if os.path.exists(adv_images_path):
                saved_data = torch.load(adv_images_path)  # Load the saved .pt file
                adv_images = saved_data["adv_images"]  # Extract adversarial images
                labels = saved_data["labels"]  # Extract labels
                print(f"Loaded adversarial examples for {attack_name} attack on {model_name} from {adv_images_path}.")

                adversarial_accuracy = evaluate_attack(model, adv_images, labels)  # Pass tensors directly
                print(
                    f"Accuracy on the test set after {attack_name} attack for {model_name}: {adversarial_accuracy:.2f}%"
                )
            else:
                print(
                    f"Adversarial examples for {attack_name} attack on {model_name} not found. Skipping accuracy calculation."
                )
                continue
