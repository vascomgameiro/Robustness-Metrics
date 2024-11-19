
import torchvision.models as models
from train import PyTorchTrainer
#mesmo nome que no https://github.com/fra31/evaluating-adaptive-test-time-defenses/tree/master

models_to_train = [
    {"name": "mobilenet_v3", "model": models.mobilenet_v3_small, "params": {"lr": 0.001}} #,
    #{"name": "resnet18", "model": models.resnet18, "params": {"lr": 0.0001}},
]

for config in models_to_train:
    model_name = f"{config['name']}_lr{config['params']['lr']}"
    trainer = PyTorchTrainer(
        model=config["model"](pretrained=True),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(config["model"](pretrained=True).parameters(), lr=config["params"]["lr"]),
    )

    trainer.train(num_epochs=10, model_name=model_name, save_dir="project_root")
    trainer.save_predictions(val_loader, f"project_root/results/{model_name}/val_predictions.csv", "val")
    trainer.save_predictions(test_loader, f"project_root/results/{model_name}/test_predictions.csv", "test")
    trainer.save_training_info(f"project_root/results/{model_name}/training_info.json")
    trainer.save_plots(f"project_root/reports/figures/{model_name}_loss_accuracy.png")
