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
        return x, y


def dataloader(path: str, batch_size: int, dataset=None):
    """
    Load CIFAR train, val, and test tensors into DataLoaders.
    """
    if dataset is None:
        train_x, train_y = torch.load(os.path.join(path, "train_cifar.pt"), weights_only=False)
        val_x, val_y = torch.load(os.path.join(path, "val_cifar.pt"), weights_only=False)
        test_x, test_y = torch.load(os.path.join(path, "test_cifar.pt"), weights_only=False)

        train_dataset = TensorDataset(train_x, train_y)
        val_dataset = TensorDataset(val_x, val_y)
        test_dataset = TensorDataset(test_x, test_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    else:
        test_x, test_y = torch.load(os.path.join(path, f"{dataset}.pt"), weights_only=False)
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return test_loader
