import torch
import os
import torchattacks
from src.data_loader import dataloader
from model_constructor import models_iterator
from attack_acc import all_attacks
from utils import (
    train_save_model,
    print_save_measures,
    get_logits_and_labels,
    load_logits_and_labels_attacks,
    calculate_metric_differences,
)
import src.measures_performance as measures_performance
import measures_norm
import measures_sharpness
from dotenv import load_dotenv


depths = [2, 4]
filters_sizes = {
    "2": [[8, 16, 160, 80], [32, 64, 320, 160]],
    "4": [[4, 8, 16, 32, 200, 200, 160, 80], [8, 16, 32, 64, 400, 320, 160, 80]],
}
lrs = [0.01, 0.001, "scheduler"]
drops = {"2": [[0.0, 0.0], [0.5, 0.2]], "4": [[0.0] * 4, [0.5, 0.3, 0.3, 0.2]]}
optimizers = ["adam", "sgd"]
models_to_train = models_iterator(depths, filters_sizes, optimizers, drops, lrs)

attacks_to_test = [
    {"name": "FGSM attack", "model": torchattacks.FGSM, "params": {"eps": 0.005}},  # estava 0.1
    {
        "name": "PGD attack",
        "model": torchattacks.PGD,
        "params": {"eps": 0.001, "alpha": 0.005, "steps": 15},
    },  # alpha 0.01
    {"name": "CW attack", "model": torchattacks.CW, "params": {"lr": 0.001, "steps": 20}},
    {"name": "DeepFool attack", "model": torchattacks.DeepFool, "params": {"overshoot": 0.02, "steps": 20}},
    {"name": "BIM attack", "model": torchattacks.BIM, "params": {"eps": 0.001, "alpha": 0.01, "steps": 20}},
    # {"name": "Square Attack", "model": torchattacks.SquareAttack, "params": {"max_queries": 10000, "eps": 0.1}},
]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")
    train_loader, val_loader, test_loader_cifar = dataloader(path_cifar=DATA_DIR, minibatch_size=32)

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

        untrained, model, train_accuracy, logits_cifar, labels_cifar = train_save_model(
            config, device, train_loader, val_loader, test_loader_cifar
        )

        # 3 - attack dataset, make prediction and save logits
        all_attacks(model, test_loader_cifar, attacks_to_test, model_name, path_to_attacks)

        # Get the logits for all datasets
        test_loader_101 = dataloader(os.path.join(DATA_DIR, "test_101"), batch_size=32, shuffle=False)
        test_loader_10c1 = dataloader(os.path.join(DATA_DIR, "test_10c1"), batch_size=32, shuffle=False)
        test_loader_10c2 = dataloader(os.path.join(DATA_DIR, "test_10c2"), batch_size=32, shuffle=False)
        test_loader_10c3 = dataloader(os.path.join(DATA_DIR, "test_10c3"), batch_size=32, shuffle=False)
        test_loader_10c4 = dataloader(os.path.join(DATA_DIR, "test_10c4"), batch_size=32, shuffle=False)
        test_loader_10c5 = dataloader(os.path.join(DATA_DIR, "test_10c5"), batch_size=32, shuffle=False)

        logits_val, labels_val = get_logits_and_labels(val_loader, model, model_name, "val", device)
        logits_test, labels_test = get_logits_and_labels(
            test_loader_cifar, model, model_name, "test", device
        )
        logits_test_101, labels_test_101 = get_logits_and_labels(
            test_loader_101, model, model_name, "test_101", device
        )
        logits_test_10c1, labels_test_10c1 = get_logits_and_labels(
            test_loader_10c1, model, model_name, "test_10c1", device
        )
        logits_test_10c2, labels_test_10c2 = get_logits_and_labels(
            test_loader_10c2, model, model_name, "test_10c2", device
        )
        logits_test_10c3, labels_test_10c3 = get_logits_and_labels(
            test_loader_10c3, model, model_name, "test_10c2", device
        )
        logits_test_10c4, labels_test_10c4 = get_logits_and_labels(
            test_loader_10c4, model, model_name, "test_10c4", device
        )
        logits_test_10c5, labels_test_10c5 = get_logits_and_labels(
            test_loader_10c5, model, model_name, "test_10c5", device
        )

        logits_fgdsa, labels_fgdsa = load_logits_and_labels_attacks("FGDSA", model_name, path_to_attacks)
        logits_fgsm, labels_fgsm = load_logits_and_labels_attacks("FGSM", model_name, path_to_attacks)
        logits_pgd, labels_pgd = load_logits_and_labels_attacks("PGD", model_name, path_to_attacks)
        logits_cw, labels_cw = load_logits_and_labels_attacks("CW", model_name, path_to_attacks)
        logits_deepfool, labels_deepfool = load_logits_and_labels_attacks(
            "DeepFool", model_name, path_to_attacks
        )
        logits_bim, labels_bim = load_logits_and_labels_attacks("BIM", model_name, path_to_attacks)
        logits_square, labels_square = load_logits_and_labels_attacks("Square", model_name, path_to_attacks)

        datasets = {
            "cifar": (logits_cifar, labels_cifar),
            "val": (logits_val, labels_val),
            "test": (logits_test, labels_test),
            "test_101": (logits_test_101, labels_test_101),
            "test_10c1": (logits_test_10c1, labels_test_10c1),
            "test_10c2": (logits_test_10c2, labels_test_10c2),
            "test_10c3": (logits_test_10c3, labels_test_10c3),
            "test_10c4": (logits_test_10c4, labels_test_10c4),
            "test_10c5": (logits_test_10c5, labels_test_10c5),
            "fgdsa": (logits_fgdsa, labels_fgdsa),
            "fgsm": (logits_fgsm, labels_fgsm),
            "pgd": (logits_pgd, labels_pgd),
            "cw": (logits_cw, labels_cw),
            "deepfool": (logits_deepfool, labels_deepfool),
            "bim": (logits_bim, labels_bim),
            "square": (logits_square, labels_square),
        }

        # 2) Evaluate the model metrics for each dataset
        complex_metrics = {}
        for name, (logits, labels) in datasets.items():
            complex_metrics[f"complexity_{name}"] = measures_performance.evaluate_model_metrics(
                logits, labels
            )

        # 3) Calculate differences relative to a baseline (e.g., "test")
        compare_against_test = [
            "test_101",
            "test_10c1",
            "test_10c2",
            "test_10c3",
            "test_10c4",
            "test_10c5",
            "fgdsa",
            "fgsm",
            "pgd",
            "cw",
            "deepfool",
            "bim",
            "square",
        ]

        complex_diff = {}
        for name in compare_against_test:
            complex_diff[f"complexity_diff_{name}"] = calculate_metric_differences(
                complex_metrics["test"], complex_metrics[name]
            )
        print_save_measures(
            complex_metrics,
            "Complex measures for Cifar test set",
            f"{path_to_measures}/complexity_cifar.pt",
        )
        print_save_measures(
            complex_diff,
            "Complexity differences for Cifar test set",
            f"{path_to_measures}/complexity_diff.pt",
        )

        model.eval()

        # Norm Measures
        print(f"Train Accuracy: {train_accuracy}")
        measures = measures_norm.calculate_generalization_bounds(
            model, untrained, train_loader, val_loader, nchannels, img_dim, device
        )
        print_save_measures(measures, "Norm Measures", f"{path_to_measures}/norm_measures.pt")

        # Sharpness Measures
        lr = config["params"]["lr"] / 100 if config["params"]["lr"] != "scheduler" else 0.00001
        if not os.path.exists(f"{path_to_measures}/sharpness.pt"):
            sharpness_metrics = measures_sharpness.calculate_combined_metrics(
                model, untrained, train_loader, train_accuracy, lr, device
            )
            print_save_measures(sharpness_metrics, "Sharpness measures", f"{path_to_measures}/sharpness.pt")


if __name__ == "__main__":
    main()
