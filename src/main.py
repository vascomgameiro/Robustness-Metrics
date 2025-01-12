import torch, os, itertools, numpy, copy, torchattacks
from torch import nn, optim
from src.data_loader import dataloader
from torchvision import datasets, transforms, models
from pytorch_trainer import PyTorchTrainer
from model_constructor import models_iterator
from attack_acc import all_attacks
from utils import train_save_model, print_save_measures
import measures_complexity
import measures_norm
import measures_sharpness
import numpy as np


depths = [2, 4]
filters_sizes = {
    "2": [[8, 16, 160, 80], [32, 64, 320, 160]],
    "4": [[4, 8, 16, 32, 200, 200, 160, 80], [8, 16, 32, 64, 400, 320, 160, 80]],}
lrs = [0.01, 0.001, "scheduler"]
drops = {"2": [[0.0, 0.0], [0.5, 0.2]], "4": [[0.0] * 4, [0.5, 0.3, 0.3, 0.2]]}
optimizers = ["adam", "sgd"]
models_to_train = models_iterator(depths, filters_sizes, optimizers, drops, lrs)

attacks_to_test = [
    {"name": "FGSM attack", "model": torchattacks.FGSM, "params": {"eps": 0.005}}, #estava 0.1
    {"name": "PGD attack", "model": torchattacks.PGD, "params": {"eps": 0.001, "alpha": 0.005, "steps": 15}}, #alpha 0.01
    {"name": "CW attack", "model": torchattacks.CW, "params":  {"lr": 0.001, "steps": 20}},
    {"name": "DeepFool attack", "model": torchattacks.DeepFool, "params": {"overshoot": 0.02, "steps": 20}},
    {"name": "BIM attack", "model": torchattacks.BIM, "params": {"eps": 0.001, "alpha": 0.01, "steps": 20}},
    #{"name": "Square Attack", "model": torchattacks.SquareAttack, "params": {"max_queries": 10000, "eps": 0.1}},
]

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1 - get data and create dataloader
    data_dir = "/Users/mariapereira/Desktop/data/cifar/processed"

    train_loader, val_loader, test_loader_cifar = dataloader(path_cifar=data_dir, minibatch_size= 32)

    images, ys = next(iter(train_loader))
    img_dim = images.shape[2]  # assuming square dimensions
    nchannels = images.shape[1]

    for config in models_to_train:
        # 2 - train model, save it and the logits

        model = config["model"]
        model_name = config["name"]
        path_to_model = f"models/{model_name}"
        path_to_measures = os.path.join(path_to_model, "measures")
        path_to_attacks = os.path.join(path_to_model, "attacks")

        untrained, model, train_accuracy, logits_cifar, labels_cifar = train_save_model(config, device, train_loader, val_loader, test_loader_cifar)
        
        # 3 - attack dataset, make prediction and save logits
        all_attacks(model, test_loader_cifar, attacks_to_test,  model_name, path_to_attacks)

        # 4 - calculate metrics
        complex_cifar = measures_complexity.evaluate_model_metrics(logits_cifar, labels_cifar)
        print_save_measures(complex_cifar, "Complex measures for Cifar test set", f"{path_to_measures}/complexity_cifar.pt")

        model.eval()# Norm Measures
        # Norm Measures
        print(f'Train Accuracy: {train_accuracy}')
        measures = measures_norm.calculate_generalization_bounds(model, untrained, train_loader, val_loader, nchannels, img_dim, device)
        print_save_measures(measures, "Norm Measures", f"{path_to_measures}/norm_measures.pt")

        # Sharpness Measures
        #if not os.path.exists(f"{path_to_measures}/sharpness.pt"):
        #    sharpness_metrics = measures_sharpness.calculate_combined_metrics(model, untrained, train_loader, train_accuracy)
        #    print_save_measures(sharpness_metrics, "Sharpness measures", f"{path_to_measures}/sharpness.pt")
        

if __name__ == "__main__":
    main()
