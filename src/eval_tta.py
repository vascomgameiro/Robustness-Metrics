
import torchvision.models as models
from train import PyTorchTrainer
from data_loader2 import dataloader
from torch import nn, optim
#mesmo nome que no https://github.com/fra31/evaluating-adaptive-test-time-defenses/tree/master

path_tiny = "/Users/clarapereira/Desktop/Uni/Ano_5/PIC/datasets/test/train.pt" #diretoria para o tensor
path_r = "/Users/clarapereira/Desktop/Uni/Ano_5/PIC/datasets/test/test.pt" #diretoria para o tensor

models_to_train = [
    {"name": "mobilenet_v3", "model": models.mobilenet_v3_small, "params": {"lr": 0.001}} #,
    #{"name": "resnet18", "model": models.resnet18, "params": {"lr": 0.0001}},
]

train_loader, val_loader, test_loader_tiny, test_loader_r = dataloader(path_tiny, path_r)

for config in models_to_train:
    model_name = f"{config['name']}_lr{config['params']['lr']}"
    trainer = PyTorchTrainer(
        model=config["model"](pretrained=True), #falta aqui criar uma função que altere alguma parte da estrutura
        train_loader=train_loader,
        val_loader=val_loader,
        # test_loader=test_loader_r, --por enquanto, o trainer ainda não pega no testset
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(config["model"](pretrained=True).parameters(), lr=config["params"]["lr"]),
    )

    trainer.train(num_epochs=1)
    #trainer.save_predictions(val_loader, f"project_root/results/{model_name}/val_predictions.csv", "val")
    #trainer.save_predictions(test_loader_r, f"project_root/results/{model_name}/test_predictions.csv", "test")
    #trainer.save_training_info(f"project_root/results/{model_name}/training_info.json")
    #trainer.save_plots(f"project_root/reports/figures/{model_name}_loss_accuracy.png")
