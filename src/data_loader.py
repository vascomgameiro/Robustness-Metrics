import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


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


def dataloader(in_distribution_path, out_distribution_path, split=False):
    """
    Load and split the in distribution and out-of-distribution datasets.
    """
    # Load datasets
    data_id = torch.load(in_distribution_path)
    data_od = torch.load(out_distribution_path)

    # Split id in train, val, test
    train_x, test_id_x, train_y, test_id_y = train_test_split(
        data_id[0], data_id[1], test_size=0.2, stratify=data_id[1], random_state=42
    )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.1, stratify=train_y, random_state=42
    )

    # Create datasets
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_id = TensorDataset(test_id_x, test_id_y)
    r_dataset = TensorDataset(data_od[0], data_od[1])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader_id = DataLoader(test_id, batch_size=32, shuffle=False)
    test_loader_r = DataLoader(r_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader_id, test_loader_r, test_id
