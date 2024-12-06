import torch
import torchvision.models as models
from torch import nn, optim

from model_mod import modify_last_layer, SimpleCNN
from data_loader import dataloader
from pytorch_trainer import PyTorchTrainer
from attacks import (
    fgsm_attack,
    pgd_attack,
    cw_attack,
    bim_attack,
    square_attack,
    deepfool_attack,
     evaluate_attack_with_logits,
)
import os


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
    #{"name": "PGD attack", "model": pgd_attack, "params": {"eps": 0.1, "alpha": 0.01, "steps": 5}},
    # {'name': 'CW attack', 'model': cw_attack, 'params': {'confidence': 10, 'steps': 3, 'lr': 0.1}},
    #{"name": "DeepFool attack", "model": deepfool_attack, "params": {"overshoot": 0.02, "max_iter": 3}},
    # {'name': 'BIM attack', 'model': bim_attack, 'params': {'eps': 0.1, 'alpha': 0.01, 'steps': 3}},
    # {'name': 'Square Attack', 'model': square_attack, 'params': {'max_queries': 10000, 'eps': 0.1}},
]


# save_dir='/models/Attacked_tensors/'


import torch
import os

save_dir='"/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/Attacked_tensors/"'


def all_attacks(model, test_loader,attacks_to_test, model_name, save_dir="/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/Attacked_tensors/"):
    for config in attacks_to_test:
        attack_name = config["name"]
        params = config["params"]
        attack_fn = config["model"]
        generate_and_save_attack_logits_with_labels(model, test_loader, attack_fn, attack_name, params, model_name, save_dir)


def generate_and_save_attack_logits_with_labels(model, test_loader, attack_fn, attack_name, params, model_name, save_dir="/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/Attacked_tensors/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_logits = []
    all_labels = []
    
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating adversarial examples and saving logits for {model_name} using {attack_name}...")

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial images
        adv_images = attack_fn(model, images, labels, **params)
        
        # Get logits for adversarial examples
        with torch.no_grad():
            logits = model(adv_images)  # Forward pass on adversarial images
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # Concatenate all logits and labels
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save the logits and labels
    save_path = os.path.join(save_dir, f"{model_name}_{attack_name}_logits_labels.pth")
    torch.save({"logits": all_logits, "labels": all_labels}, save_path)

    print(f"Saved logits and labels to {save_path}")
    #AQUI SE QUISERMOS PODEMOS POR AS OUTRAS METRICAS MAIS LOUCAS PQ TEMOS OS LOGITS
    adversarial_accuracy = evaluate_attack_with_logits(all_logits, all_labels)  # Pass logits and labels
    print(f"Accuracy on the test set after {attack_name} attack for {model_name}: {adversarial_accuracy:.2f}%")   
 
# def evaluate_attacks_with_logits_and_labels(
#     models_to_train,
#     attacks_to_test, 
#     test_loader_tiny,
#     save_dir):
#     """
#     Evaluate all attacks on all models using saved logits and labels, and calculate adversarial accuracy.
#     """

#     for model_config in models_to_train:
#         model_name = model_config["name"]
#         model_class = model_config["model"]
#         model_params = model_config["params"]

#         model = model_class()
# #         print(model_path)
        # model_path = f"/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/{model_name}/{model_name}.pt"
        # print(model_path)
        # try:
        #     model.load_state_dict(torch.load(model_path))  # Load pre-trained weights
        #     print(f"Loaded pre-trained weights for {model_name} from {model_path}.")
        # except FileNotFoundError:
        #     print(f"No pre-trained weights found for {model_name}, skipping.")
            # continue

#         print(f"Evaluating attacks on {model_name}...")
#         for config in attacks_to_test:
#             attack_name = config["name"]
#             params = config["params"]
#             attack_fn = config["model"]

            # generate_and_save_attack_logits_with_labels(model, test_loader_tiny, attack_fn, attack_name, params, model_name, save_dir):
    
            # logits_labels_path =  os.path.join(save_dir, f"{model_name}_{attack_name}_logits_labels.pth")

            # if os.path.exists(logits_labels_path):
            #     saved_data = torch.load(logits_labels_path)  # Load the saved logits and labels
            #     saved_logits = saved_data["logits"]
            #     saved_labels = saved_data["labels"]

            #     print(f"Loaded logits and labels for {attack_name} attack on {model_name} from {logits_labels_path}.")

            #     adversarial_accuracy = evaluate_attack(saved_logits, saved_labels)  # Pass logits and labels
            #     print(
            #         f"Accuracy on the test set after {attack_name} attack for {model_name}: {adversarial_accuracy:.2f}%"
            #     )
            # else:
            #     print(
            #         f"Logits and labels for {attack_name} attack on {model_name} not found. Skipping accuracy calculation."
            #     )
            #     continue