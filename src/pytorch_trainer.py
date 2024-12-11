import os

import matplotlib.pyplot as plt
import numpy as np
import torch, copy


class PyTorchTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device="cpu"):
        """
        Initialize a PyTorchTrainer object.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be trained.
        train_loader : torch.utils.data.DataLoader
            The DataLoader for the training set.
        val_loader : torch.utils.data.DataLoader
            The DataLoader for the validation set.
        criterion : torch.nn.Module
            The loss function to be used for training.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training.
        scheduler : torch.optim.lr_scheduler, optional
            The learning rate scheduler to be used for training. If None, no scheduler is used.
        device : str, optional
            The device to be used for training. If "cuda", training is done on the GPU and if "cpu", training is done on the CPU.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = model.to(device)
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0
        self.best_model = None

    def train(self, num_epochs=10, early_stopping_patience=5):
        print("Starting Training...\n")
        no_improvement_epochs = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()

            # Adjust learning rate if scheduler is used
            if self.scheduler:
                self.scheduler.step()

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Epoch [{epoch+1}/{num_epochs}]: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save the best model based on validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model = copy.deepcopy(self.model)  # Save best model weights
                print(f"New best model found! Validation Accuracy: {val_acc:.2f}%")
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # Early stopping 
            if no_improvement_epochs >= early_stopping_patience:
                self.model = self.best_model
                print(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
                break

        print("Training Complete!")

    def _train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, 100.0 * correct / total

    def _validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def save_best_model(self, path="best_model.pt"):
        saved = self.best_model
        if saved is not None:
            torch.save(saved.state_dict(), path)
            print(f"Best model saved to {path}")
        else:
            print("No model was saved because no improvement was detected.")

    def load_model(self, path="model.pt"):
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)
        print(f"Model loaded from {path}")

    def predict(self, data_loader):
        self.model.eval()
        logits = []
        y = []
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                logits.extend(outputs.cpu().numpy()) 
                y.extend(labels.cpu().numpy())
        return np.array(logits), np.array(y)

    def save_predictions(self, predictions, path="predictions.npy"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, predictions)
        print(f"Predictions saved to {path}")

    def save_plots(self, path):
        
        os.makedirs(path, exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        # Loss Plot
        plt.figure()
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Over Epochs")
        plt.legend()
        loss_path = os.path.join(path, "loss.png")
        plt.savefig(loss_path)
        print(f"Loss plot saved to {loss_path}")
        plt.close()

        # Accuracy Plot
        plt.figure()
        plt.plot(epochs, self.history["train_acc"], label="Train Accuracy")
        plt.plot(epochs, self.history["val_acc"], label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Over Epochs")
        plt.legend()
        acc_path = os.path.join(path, "acc.png") 
        plt.savefig(acc_path)
        print(f"Accuracy plot saved to {acc_path}")
        plt.close()
