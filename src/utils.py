import torch
import os
import copy
import numpy
from torch import nn
from pytorch_trainer import PyTorchTrainer


def train_save_model(config, device, train_loader, val_loader, test_loader_cifar): 
    model = config["model"]
    model_name = config["name"]
    optimizer = config["params"]["optimizer"]
    lr = config["params"]["lr"]

    untrained = copy.deepcopy(model)
    # the current working dir should be the project root: robustness-metrics
    path_to_model = f"models/{model_name}"
    path_to_predictions = os.path.join(path_to_model, "predictions")
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
        train_accuracy = trainer.final_train_acc
        trainer.save_best_model(path_to_model)
        trainer.save_plots(path_to_plots)

        logits_cifar, labels_cifar = trainer.predict(test_loader_cifar)
        trainer.save_predictions(logits_cifar, f"{path_to_predictions}/cifar.npy")
        trainer.save_accuracies(path_to_predictions)

    else:
        print(f"Model {model_name} is already trained. Skipping training step.")

        model.load_state_dict(torch.load(f"{path_to_model}/trained.pt", map_location = device))
        untrained.load_state_dict(torch.load(f"{path_to_model}/untrained.pt", map_location = device))

        model = model.to(device) 
        untrained = untrained.to(device) 

        accuracies_dic = torch.load(f"{path_to_predictions}/accuracies.pt",  weights_only=False)
        train_accuracy = accuracies_dic["train_acc"]
        
        logits_cifar = numpy.load(f"{path_to_predictions}/cifar.npy")
        labels_cifar = []
        for _, labels in test_loader_cifar:
            labels_cifar.extend(labels.numpy())

    return untrained, model, train_accuracy, logits_cifar, labels_cifar


def print_save_measures(dic, statement, path_save):
    print("\n")
    print(statement)
    for key, value in dic.items():
        print(f"{key}: {value}")
    print(f"Dictionary saved to {path_save}")
    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    torch.save(dic, path_save)