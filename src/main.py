import torch
import utils
import torchvision.models as models
from torch import nn, optim
import os

from model_mod import modify_last_layer, SimpleCNN
from data_loader import dataloader
from pytorch_trainer import PyTorchTrainer
import attack_acc



# mesmo nome que no https://github.com/fra31/evaluating-adaptive-test-time-defenses/tree/master

models_to_train = [
        {"name": "simple_cnn", "model": SimpleCNN, "params": {"lr": 0.001}}
        # {"name": "mobilenet_v3", "model": models.mobilenet_v3_small, "params": {"lr": 0.001}}  # ,
        # {"name": "resnet18", "model": models.resnet18, "params": {"lr": 0.0001}},
    ]

attacks_to_test = [
    {"name": "FGSM attack", "model": fgsm_attack, "params": {"eps": 0.1}},
    {"name": "PGD attack", "model": pgd_attack, "params": {"eps": 0.1, "alpha": 0.01, "steps": 1}},
]


######
# queremos:
#1 - ir buscar dados e criar dataloader
#2 - treinar modelo (guardar o modelo e os logits!!)
#3 - fazer ataques e guardar (guardar o que exatamente??)
#4 - calcular as métricas!

def main():

    #utils.configure_seed(seed=42)

    #1 - ir buscar dados e criar dataloader
    data_dir = "..."
    path_tiny = os.path.join(data_dir, "tiny.pt")  # diretoria para o tensor
    path_r = os.path.join(data_dir, "r.pt")  # diretoria para o tensor

    train_loader, val_loader, test_loader_tiny, test_loader_r, test_tiny = dataloader(path_tiny, path_r)

    #2 - treinar modelo (guardar o modelo e os logits!!)
    
    for config in models_to_train:
        model_name = f"{config['name']}_lr{config['params']['lr']}"
        model = config["model"]()

        """
        if the model is pretrained, need to modify its last layer to include nr of classes
        model=modify_last_layer(
            # model, model_name, len(torch.unique(train_loader.dataset.labels))),  # falta aqui criar uma função que altere alguma parte da estrutura
            model=model,
            train_loader=train_loader,
        """
        #the current working dir should be the project root: robustness-metrics

        path_to_model = f"models/{model_name}"
        path_to_predictions = os.path.join(path_to_model, "predictions")
        path_to_plots = os.path.join(path_to_model, f"plots/{model_name}")

        trainer = PyTorchTrainer(
            model = model,
            train_loader = train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(model.parameters(), lr=config["params"]["lr"]),
        )

        trainer.train(num_epochs=1)
        trainer.save_best_model(f"{path_to_model}/{model_name}.pt")
        trainer.save_predictions(trainer.predict(test_loader_r), f"{path_to_predictions}/r.npy")
        trainer.save_plots(path_to_plots)

    #3 - fazer ataques e guardar(cada ataque é guardado dentro da pasta do modelo correspondente, dentro da pasta "attacks")
    #     e avaliar as accuracies
    
    attack_acc.evaluate_and_save_all_attacks(models_to_train, test_loader_tiny, save_dir = "models")

    
    #4 - calcular as métricas!

    


if __name__ == '__main__':
    main()
