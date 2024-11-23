import torch
import torchvision.models as models
from torch import nn, optim

from model_mod import modify_last_layer, SimpleCNN
from data_loader import load_datasets
from train import PyTorchTrainer
import torchattacks
from attacks import fgsm_attack ,pgd_attack, evaluate_attack


# mesmo nome que no https://github.com/fra31/evaluating-adaptive-test-time-defenses/tree/master

path_tiny = "/Users/joanacorreia/Desktop/AECD/test/train.pt"  # diretoria para o tensor
path_r = "/Users/joanacorreia/Desktop/AECD/test/test.pt"  # diretoria para o tensor

models_to_train = [  {'name': 'simple_cnn','model': SimpleCNN, 'params': {'lr': 0.001}}
   # {"name": "mobilenet_v3", "model": models.mobilenet_v3_small, "params": {"lr": 0.001}}  # ,
    # {"name": "resnet18", "model": models.resnet18, "params": {"lr": 0.0001}},
]

attacks_to_test = [{'name': 'FGSM attack', 'model': fgsm_attack, 'params': {'eps':0.1} },
                   {'name': 'PGD attack', 'model': pgd_attack, 'params': {'eps':0.1, 'alpha': 0.01, 'steps':1} }]

train_loader, val_loader, test_loader_tiny, test_loader_r= load_datasets(path_tiny, path_r)

for config in models_to_train:
    model_name = f"{config['name']}_lr{config['params']['lr']}"
    model = config["model"]()
    
    trainer = PyTorchTrainer(
        #model=modify_last_layer(
            #model, model_name, len(torch.unique(train_loader.dataset.labels))),  # falta aqui criar uma função que altere alguma parte da estrutura
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        # test_loader=test_loader_r, --por enquanto, o trainer ainda não pega no testset
        criterion=nn.CrossEntropyLoss(),
        #optimizer=optim.Adam(config["model"](pretrained=True).parameters(), lr=config["params"]["lr"]),
        optimizer=optim.Adam(model.parameters(), lr=config["params"]["lr"]),
    )

    trainer.train(num_epochs=1)
    for config in attacks_to_test:
       
        #adversarial_accuracy = attack_test(model, test_loader_tiny, torchattacks.FGSM(model, eps=0.1)  )  # depois tem de ser definido como attack e serem rodados.
        adversarial_accuracy = evaluate_attack(model, test_loader_tiny,config["model"], **config['params'])
        #adversarial_accuracy = attack_test(model, test_loader_tiny,attack = torchattacks.DeepFool(model, steps=50))
        print(f'Accuracy on the test set after ',config['name'], ' attack: {adversarial_accuracy}%')
        print(f"Accuracy on the test set after {config['name']} attack: {adversarial_accuracy:.2f}%")
        print("Adversarial Accuracy:", adversarial_accuracy)  # Debug

    
    # trainer.save_predictions(val_loader, f"project_root/results/{model_name}/val_predictions.csv", "val")
    # trainer.save_predictions(test_loader_r, f"project_root/results/{model_name}/test_predictions.csv", "test")
    # trainer.save_training_info(f"project_root/results/{model_name}/training_info.json")
    # trainer.save_plots(f"project_root/reports/figures/{model_name}_loss_accuracy.png")
    # trainer.save_plots(f"project_root/reports/figures/{model_name}_loss_accuracy.png")
    # trainer.save_plots(f"project_root/reports/figures/{model_name}_loss_accuracy.png")
