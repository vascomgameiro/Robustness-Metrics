import torchattacks
import torch
from torch.utils.data import DataLoader
from data_loader import TensorDataset
 #exemplos de bibliotecas 

# (we'll use the PGD attack as an example)

#attack = torchattacks.PGD(model, eps=0.1, alpha=0.01, steps=40)

import torch
import torch.nn as nn

# FGSM Attack Implementation
def fgsm_attack(model, images, labels, eps):
 
    device = torch.device("cpu")
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True

    
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    # Generate adversarial examples
    adv_images = images + eps * images.grad.sign()

    # Clip the adversarial images to stay within valid data range
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images

def pgd_attack(model, images, labels, eps, alpha, steps):
    device = torch.device("cpu")
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    ori_images = images.clone().detach()

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        perturbation = torch.clamp(adv_images - ori_images, -eps, eps)
        images = torch.clamp(ori_images + perturbation, 0, 1).detach()

    return images


def evaluate_attack(model, test_loader, attack, **attack_params):
    """
    Evaluate the model accuracy on adversarial examples generated using any attack.
    
    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set.
        attack: The attack function to generate adversarial examples.
        **attack_params: Additional parameters for the attack function.
    
    Returns:
        accuracy: The model's accuracy on adversarial examples.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    device = torch.device("cpu")
    for images, labels in test_loader:
        
        images, labels = images.to(device), labels.to(device)

        adv_images = attack(model, images, labels, **attack_params)

        # Make predictions on adversarial examples
        outputs = model(adv_images)
        _, predicted = outputs.max(1)

        # Update the correct predictions count
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Calculate accuracy
    accuracy = correct / total * 100
    return accuracy
