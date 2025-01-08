import torch, os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class TensorDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        """
        Initialize the TensorDataset with features and labels.

        Parameters:
            data_path (str): Path to the .pt file containing preprocessed data.
                            The file is expected to contain a tuple (features, labels).
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def dataloader(path_cifar, minibatch_size, transform = None):
    """
    Load the train, val and test cifar tensors
    """
    # Load datasets
    path_train = os.path.join(path_cifar, "train_cifar.pt")
    path_val = os.path.join(path_cifar, "val_cifar.pt")
    path_test = os.path.join(path_cifar, "test_cifar.pt")

    train_x, train_y = torch.load(path_train, weights_only=False)
    val_x, val_y = torch.load(path_val, weights_only=False)
    test_cifar_x, test_cifar_y = torch.load(path_test, weights_only=False)

    # Create datasets
    train_dataset = TensorDataset(train_x, train_y, transform)
    val_dataset = TensorDataset(val_x, val_y, transform)
    test_tiny = TensorDataset(test_cifar_x, test_cifar_y, transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=True)
    test_loader_tiny = DataLoader(test_tiny, batch_size=minibatch_size, shuffle=False)


    return train_loader, val_loader, test_loader_tiny


