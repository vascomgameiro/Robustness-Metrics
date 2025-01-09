import torch
import os
from torch.utils.data import Dataset, DataLoader


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


def dataloader(path_cifar, minibatch_size, transform=None):
    """
    Load CIFAR train, val, and test tensors into DataLoaders.
    """
    train_x, train_y = torch.load(os.path.join(path, "train_cifar.pt"), weights_only=False)
    val_x, val_y = torch.load(os.path.join(path, "val_cifar.pt"), weights_only=False)
    test_x, test_y = torch.load(os.path.join(path, "test_cifar.pt"), weights_only=False)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    test_ds = TensorDataset(test_x, test_y)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=True)
    test_loader_tiny = DataLoader(test_tiny, batch_size=minibatch_size, shuffle=False)

    return train_dl, val_dl, test_dl
