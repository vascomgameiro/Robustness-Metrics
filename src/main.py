import torch, os, itertools
import torchvision.models as models
import attack_acc, adversarial_attacks
from attack_acc import all_attacks
import model_constructor as constructor
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


#https://github.com/fra31/evaluating-adaptive-test-time-defenses/tree/master

# possible optimizers: {SGD, ADAM, RMSprop}
#atenção a early stopping: nas primeiras epochs não faz sentido
#some pretrained models: {MobileNetV2, ResNet50, EfficientNet}

def models_iterator(nr_conv, filters, maxpool_batchnorm, nr_fconnected, fc_sizes, act_funs, drops, lr):
    models_to_train = []
    configurations = list(itertools.product(maxpool_batchnorm, drops))
    for nr_c in range(nr_conv):
        for nr_fc in range(1, nr_fconnected):
            for config in configurations:

                maxpool_batchnorm, drops = config

                if not(nr_fc == 1 and drops == [0]*3): #avoid making 2 copied models in the case where there is only 1 layer: both "drops" options will be the same [] !! 
                    nr_filters = filters[:nr_c]

                    conv_layers = constructor.Conv(nr_conv=nr_c, nr_filters=nr_filters, maxpool_batchnorm=maxpool_batchnorm)

                    fc_size=fc_sizes[4-nr_fc:] #want to slice from the end
                    act_fun=act_funs[:nr_fc-1]
                    dropouts = drops[4-nr_fc:] #want to slice from the end
                    
                    fc_layers = constructor.FC(nr_fc=nr_fc, fc_size=fc_size, act_funs=act_fun, dropouts=dropouts, in_features=conv_layers.finaldim)
                    
                    # Create the model using the CNN constructor
                    model = constructor.CNN(conv_layers=conv_layers, fc_layers=fc_layers, num_classes=62)  # Assuming 62 classes as an example
                    
                    # Store the model and its parameters in the list
                    model_info = {
                        "name": f"{model.name}", 
                        "model": model,
                        "params": {"lr": lr}
                    }
                    
                    models_to_train.append(model_info)

    return models_to_train


filters = [8, 16, 32, 64]
maxpool_batchnorm = ["False", "True"]
fc_sizes = [320, 160, 80, 62]
lr = 0.01
act_funs = ['ReLU'] * 3
drops =[ [0.5, 0.2, 0.2], [0]*3]

#models_to_train = models_iterator(4, filters, maxpool_batchnorm, 5, fc_sizes, act_funs, drops, lr)
#print(f"list of {len(models_to_train)} models generated!!")
#print(models_to_train)


#try this!!
decent_conv = constructor.Conv(3, [16, 32, 64])
decent_fc = constructor.FC(3, [300, 150, 62], ['ReLU']* 2, dropouts= [0.5, 0.2], in_features=decent_conv.finaldim)
decent_model = constructor.CNN(decent_conv, decent_fc, 62, 0.01)

models_to_train = [
        {"name": decent_model.name, "model": decent_model}
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


"""
        if the model is pretrained, need to modify its last layer to include nr of classes
        maybe do this outside
        model=modify_last_layer(
            # model, model_name, len(torch.unique(train_loader.dataset.labels))),  # falta aqui criar uma função que altere alguma parte da estrutura
            model=model,
            train_loader=train_loader,
        """

######
# queremos:
#1 - ir buscar dados e criar dataloader
#2 - treinar modelo (guardar o modelo e os logits!!)
#3 - fazer ataques e guardar (guardar o que exatamente??)
#4 - calcular as métricas!

def main():

    #1 - ir buscar dados e criar dataloader
    data_dir = "/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/data"
    path_tiny = os.path.join(data_dir, "tiny.pt")  # diretoria para o tensor
    path_r = os.path.join(data_dir, "r.pt")  # diretoria para o tensor

    train_loader, val_loader, test_loader_tiny, test_loader_r, test_tiny = dataloader(path_tiny, path_r)

    #2 - treinar modelo, guardar o modelo e os logits
    
    for config in models_to_train:
        
        model = config["model"]
        model_name = config["name"]

        #the current working dir should be the project root: robustness-metrics
        path_to_model = f"models/{model_name}"
        path_to_predictions = os.path.join(path_to_model, "predictions")
        path_to_plots = os.path.join(path_to_model, f"plots/{model_name}")

        trainer = PyTorchTrainer(
            model = model,
            train_loader = train_loader,
            val_loader=val_loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(model.parameters(), lr=model.lr),
        )

        os.makedirs(path_to_model, exist_ok=True)
        torch.save(model.state_dict(), f"{path_to_model}/{model_name}_untrained.pt")
        trainer.train(num_epochs=1)
        trainer.save_best_model(f"{path_to_model}/{model_name}.pt")
        trainer.save_predictions(trainer.predict(test_loader_r), f"{path_to_predictions}/r.npy")
        trainer.save_plots(path_to_plots)
        #falta aqui avaliar acc com o test normal
    
    #3 - fazer ataques e guardar(cada ataque é guardado dentro da pasta do modelo correspondente, dentro da pasta "attacks")
    #     e avaliar as accuracies
    
        all_attacks(model, test_loader_r,attacks_to_test,  model_name, save_dir="/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/Attacked_tensors/")
    
    
    #4 - calcular as métricas!

    


if __name__ == '__main__':
    main()
