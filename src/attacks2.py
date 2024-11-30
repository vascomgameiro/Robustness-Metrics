import torchattacks
import torch
from torch.utils.data import DataLoader
from data_loader import TensorDataset
 



#tem ataques e a funççao evaluate

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

def bim_attack(model, images, labels, eps, alpha, steps):
    device = torch.device("cpu")
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for step in range(steps):
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Update adversarial images using gradient
        adv_images = adv_images + alpha * adv_images.grad.sign()

        # Clip the adversarial images to stay within the epsilon range
        adv_images = torch.clamp(adv_images, min=images - eps, max=images + eps)
        adv_images = torch.clamp(adv_images, 0, 1).detach_()
        adv_images.requires_grad = True

    return adv_images

def cw_attack(model, images, labels, confidence, steps, lr):
    device = torch.device("cpu")
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    optimizer = torch.optim.Adam([adv_images], lr=lr)

    for step in range(steps):
        outputs = model(adv_images)
        target_onehot = torch.zeros_like(outputs).scatter_(1, labels.unsqueeze(1), 1)
        real = (outputs * target_onehot).sum(dim=1)
        other = (outputs * (1 - target_onehot) - target_onehot * 1e4).max(dim=1)[0]

        # Loss for Carlini & Wagner
        loss = torch.clamp(other - real + confidence, min=0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clip adversarial images to valid range
        adv_images = torch.clamp(adv_images, 0, 1).detach_()
        adv_images.requires_grad = True

    return adv_images

def deepfool_attack(model, images, labels, overshoot, max_iter):
    device = torch.device("cpu")
    images = images.clone().detach().to(device)
    adv_images = images.clone().detach()

    outputs = model(adv_images).detach()
    _, preds = outputs.max(1)

    iteration = 0
    while preds.eq(labels).all() and iteration < max_iter:
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Calculate perturbation
        grad = adv_images.grad
        perturbation = grad.sign() * (overshoot / grad.norm())

        # Update adversarial images
        adv_images = adv_images + perturbation
        adv_images = torch.clamp(adv_images, 0, 1).detach_()
        adv_images.requires_grad = True

        outputs = model(adv_images).detach()
        _, preds = outputs.max(1)
        iteration += 1

    return adv_images



def square_attack(model, images, labels, eps, max_queries):
    device = torch.device("cpu")
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    adv_images = images.clone().detach()

    for _ in range(max_queries):
        # Add random noise perturbation
        noise = torch.rand_like(adv_images) * 2 * eps - eps
        adv_images = adv_images + noise
        adv_images = torch.clamp(adv_images, 0, 1)

        outputs = model(adv_images)
        preds = outputs.max(1)[1]
        if not preds.eq(labels).all():
            break

    return adv_images



def evaluate_attack(model, adv_images, labels):
    """
    Evaluate model accuracy on adversarial examples.
    """
    model.eval()
    device = next(model.parameters()).device  
    adv_images, labels = adv_images.to(device), labels.to(device)


    outputs = model(adv_images)
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / labels.size(0)
    return accuracy