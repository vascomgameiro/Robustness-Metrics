import matplotlib.pyplot as plt
import torch


class PyTorchTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0  # To track the best validation accuracy
        self.best_model = None  # To store the best model

    def train(self, num_epochs=1, early_stopping_patience=5):
        print("Starting Training...\n")
        no_improvement_epochs = 0  # To track epochs with no improvement
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
                self.best_model = self.model.state_dict()  # Save best model weights
                print(f"New best model found! Validation Accuracy: {val_acc:.2f}%")
                self.save_model() #depois definir aqui qual é o path onde queremos guardar!!
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # Early stopping
            if no_improvement_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
                break

        print("Training Complete!")

    def save_best_model(self, path="best_model.pt"):
        if self.best_model is not None:
            torch.save(self.best_model, path)
            print(f"Best model saved to {path}")
        else:
            print("No model was saved because no improvement was detected.")

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

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

    def save_model(self, path="model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="model.pt"):
        self.model.load_state_dict(torch.load(path))
        self.model = self.model.to(self.device)
        print(f"Model loaded from {path}")

    def plot_history(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Over Epochs")
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], label="Train Accuracy")
        plt.plot(epochs, self.history["val_acc"], label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Over Epochs")
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.show()
        plt.show()
