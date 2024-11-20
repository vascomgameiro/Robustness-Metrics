import torch
from torch.utils.data import DataLoader, Dataset


class TensorDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize the TensorDataset with features and labels.

        Parameters:
            data_path (str): Path to the .pt file containing preprocessed data.
                            The file is expected to contain a tuple (features, labels).
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(processed_path):
    """
    Load datasets from .pt files and split into train, val, test sets. Create
    DataLoaders for each set.

    Parameters:
        path_tiny (str): Path to the .pt file containing preprocessed tiny-imagenet data.
        path_r (str): Path to the .pt file containing preprocessed imagenet-r data.

    Returns:
        tuple: A tuple of four DataLoaders (train, val, test_tiny, test_r)
    """
    train_tiny_path = processed_path / "tiny_train.pt"
    val_tiny_path = processed_path / "tiny_val.pt"
    test_tiny_path = processed_path / "tiny_test.pt"
    r_path = processed_path / "r.pt"

    # Create datasets
    train_dataset = TensorDataset(*torch.load(train_tiny_path))
    val_dataset = TensorDataset(*torch.load(val_tiny_path))
    test_tiny = TensorDataset(*torch.load(test_tiny_path))
    r_dataset = TensorDataset(*torch.load(r_path))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader_tiny = DataLoader(test_tiny, batch_size=32, shuffle=False)
    test_loader_r = DataLoader(r_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader_tiny, test_loader_r
