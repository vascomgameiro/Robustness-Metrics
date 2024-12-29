import torch, os,, numpy, copy
from torch import nn
from data_loader_cifar import dataloader
from pytorch_trainer import PyTorchTrainer
from vgg import VGG
import measures_complexity
import measures_norm


def print_save_measures(dic, statement, path_save):
    print(f"Dictionary saved to {path_save}")
    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    torch.save(dic, path_save)

    print(statement)
    for key, value in dic.items():
        print(f"{key}: {value}")
    print("\n")


# vg11 from scratch! and adapted to the image size
# vgg_11_small = VGG(10, 0.5)

# models_to_train = []
# lrs = [0.01, 0.001, "scheduler"]
# optimizers = ["sgd", "adam"]
# for lr in lrs:
#     for optimizer in optimizers:
#         model_name = f"vgg_cifar{lr}{optimizer}"
#         model_info = {"name": model_name, "model": vgg_11_small, "params": {"lr": lr, "optimizer": optimizer}}
#         models_to_train.append(model_info)

# # print(models_to_train)
# print(f"list of {len(models_to_train)} models generated!!")

# models_to_train = [{"name": , "model": , "params": {"lr": , "optimizer": , }}]

# try this!!
# decent_conv = constructor.Conv(3, [16, 32, 64])
# decent_fc = constructor.FC(3, [300, 150, 62], ['ReLU']* 2, dropouts= [0.5, 0.2], in_features=decent_conv.finaldim)
# decent_model = constructor.CNN(decent_conv, decent_fc, 62, 0.01)

# attacks_to_test = [
#   {"name": "FGSM_attack", "model": fgsm_attack, "params": {"eps": 0.1}},
# {"name": "PGD_attack", "model": pgd_attack, "params": {"eps": 0.1, "alpha": 0.01, "steps": 5}},
# {'name': 'CW_attack', 'model': cw_attack, 'params': {'confidence': 10, 'steps': 3, 'lr': 0.1}},
# {"name": "DeepFool_attack", "model": deepfool_attack, "params": {"overshoot": 0.02, "max_iter": 3}},
# {'name': 'BIM_attack', 'model': bim_attack, 'params': {'eps': 0.1, 'alpha': 0.01, 'steps': 3}},
# {'name': 'Square_Attack', 'model': square_attack, 'params': {'max_queries': 10000, 'eps': 0.1}},
# ]

######
# queremos:
# 1 - ir buscar dados e criar dataloader
# 2 - treinar modelo (guardar o modelo e os logits!!)
# 3 - fazer ataques e guardar (guardar o que exatamente??)
# 4 - calcular as métricas!


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1 - ir buscar dados e criar dataloader
    data_dir = "data/cifar/processed"

    train_loader, val_loader, test_loader_cifar = dataloader(data_dir)

    images, ys = next(iter(train_loader))
    img_dim = images.shape[2]  # assuming square dimensions
    nchannels = images.shape[1]

    # 2 - treinar modelo, guardar o modelo e os logits

    for config in models_to_train:
        print(config)
        model = config["model"]
        model_name = config["name"]
        optimizer = config["params"]["optimizer"]
        lr = config["params"]["lr"]
        if lr == "scheduler":
            scheduler = torch.optim.lr_scheduler.ReducedLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
        else:
            scheduler = None
        untrained = copy.deepcopy(model)
        # the current working dir should be the project root: robustness-metrics
        path_to_model = f"models/{model_name}"
        path_to_predictions = os.path.join(path_to_model, "predictions")
        path_to_plots = os.path.join(path_to_model, "plots")
        path_to_measures = os.path.join(path_to_model, "measures")
        path_to_attacks = os.path.join(path_to_model, "attacks")

        if not os.path.exists(path_to_model):
            optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
            optim_cls = optims[optimizer]

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

            trainer.train(num_epochs=100, early_stopping_patience=50)
            model = trainer.best_model
            trainer.save_best_model(f"{path_to_model}/trained.pt")
            trainer.save_plots(path_to_plots)

            # logits_r, labels_r = trainer.predict(test_loader_r)
            # trainer.save_predictions(logits_r, f"{path_to_predictions}/r.npy")
            logits_cifar, labels_cifar = trainer.predict(test_loader_cifar)
            trainer.save_predictions(logits_cifar, f"{path_to_predictions}/cifar.npy")

        else:
            print(f"Model {model_name} is already trained. Skipping training step.")

            model.load_state_dict(torch.load(f"{path_to_model}/trained.pt"))
            untrained.load_state_dict(torch.load(f"{path_to_model}/untrained.pt"))
            # logits_r = numpy.load(f"{path_to_predictions}/r.npy")
            # labels_r = []
            # for _, labels in test_loader_r:
            #    labels_r.extend(labels.numpy())

            logits_cifar = numpy.load(f"{path_to_predictions}/cifar.npy")
            labels_cifar = []
            for _, labels in test_loader_cifar:
                labels_cifar.extend(labels.numpy())

        # 3 - fazer ataques e guardar(cada ataque é guardado dentro da pasta do modelo correspondente, dentro da pasta "attacks")
        #     e avaliar as accuracies

        # all_attacks(model, test_loader_r, attacks_to_test,  model_name, path_to_attacks)

        # 4 - calcular as métricas!

        # Complex Measures

        # nos datasets original/ com shift natural:
        # complex_r = measures_complexity.evaluate_model_metrics(logits_r, labels_r)
        # print_save_measures(complex_r, "Complex measures for R test set", f"{path_to_measures}/complexity_r.pt")
        complex_cifar = measures_complexity.evaluate_model_metrics(logits_cifar, labels_cifar)
        print_save_measures(
            complex_cifar, "Complex measures for Cifar test set", f"{path_to_measures}/complexity_cifar.pt"
        )

        """
        #nos ataques:
        for config in attacks_to_test:
            attack_name = config["name"]
            save_path = os.path.join(path_to_attacks, f"{attack_name}_logits_labels.pth")
            dic_attacks = torch.load(save_path)
            logits_a = dic_attacks["logits"].numpy()
            labels_a = dic_attacks["labels"].numpy()
            complex_attack = measures_complexity.evaluate_model_metrics(logits_a, labels_a)
            print_save_measures(complex_attack, f"Complex measures for attacked tiny test set, with attack {attack_name}", f"{path_to_measures}/complexity_{attack_name}.pt") 
        """

        # Norm Measures - they have nothing to do with the test set!
        measures, bounds = measures_norm.calculate_generalization_bounds(
            model, untrained, train_loader, val_loader, nchannels, img_dim, device
        )
        print_save_measures(measures, "Norm Measures", f"{path_to_measures}/norm_measures.pt")
        print_save_measures(bounds, "Norm measures: bounds", f"{path_to_measures}/norm_bounds.pt")


if __name__ == "__main__":
    main()
