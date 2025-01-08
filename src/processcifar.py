import time
from pathlib import Path

import numpy as np
import pickle
import os
import torch
import pathlib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
RAW_DIR = Path("/Users/clarapereira/Desktop/Uni/Ano_5/PIC/data/raw")
PROCESSED_DIR = Path("/Users/clarapereira/Desktop/Uni/Ano_5/PIC/data/processed")


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dict = {key.decode('utf-8'): value if not isinstance(value, bytes) else value.decode('utf-8') for key, value in dict.items()}
    return dict

def load_images(pkl_file):
    
    try:
        dict = unpickle(pkl_file)
        images = dict["data"]
        images = dict["data"]
        labels = dict["labels"]
        images = images.reshape((-1, 3, 32, 32))
        images = images.astype(np.float32) / 255.0
        print(images.shape)
    except Exception as e:
        print(f"Error processing {pkl_file}: {e}")
    return images, labels

def get_data(cifar_dir, dataset):
     
    data_path = os.path.join(cifar_dir, dataset)

    if dataset == 'cifar':
        train_images, train_labels = [], []
        for filename in os.listdir(data_path):
            if "_batch" in filename:
                pkl_file = os.path.join(data_path, filename)
                print("Loading: ", filename)
                images, labels = load_images(pkl_file)
                train_images.extend(images)
                train_labels.extend(labels)

        np_images = np.array(train_images)
        np_labels = np.array(train_labels)
    
    elif dataset == 'cifar10.1':

        label_filename = 'cifar10.1_v6_labels.npy'
        imagedata_filename = 'cifar10.1_v6_data.npy'
        label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
        images_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))

        print('Loading labels from file {}'.format(label_filepath))
        np_labels = np.load(label_filepath, allow_pickle=True)
        print('Loading image data from file {}'.format(images_filepath))
        np_images = np.load(images_filepath, allow_pickle=True)

    return np_images, np_labels

def to_tensor(images, labels):
    return (
        torch.tensor(images, dtype=torch.float32).view(-1, 3, 32, 32),
        torch.tensor(labels, dtype=torch.long),
    )


def process_data(CIFAR_DIR, PROCESSED_DIR, dataset = ''):
    start = time.time()
    print(f"Processing dataset {dataset}...")
    cifar_images, cifar_labels = get_data(CIFAR_DIR, dataset)
    end = time.time()
    print(f"Dataset {dataset} data shape: {cifar_images.shape} ({end - start:.2f} seconds)")
    start = end

    cifar_tensor, cifar_labels_tensor = to_tensor(cifar_images, cifar_labels)

    if dataset != 'cifar10':
        torch.save((cifar_tensor, cifar_labels_tensor), f"{PROCESSED_DIR}/{dataset}.pt")
        print(f"{dataset} Tensor Saved.")

    elif dataset == 'cifar10':
        torch.save((cifar_tensor, cifar_labels_tensor), PROCESSED_DIR / "cifar.pt")
        print("Cifar Tensor Saved.")

        #go from 60000 to 30000 images
        cifar_tensor, unused_tensor, cifar_labels, unused_lbs = train_test_split(
            cifar_tensor, cifar_labels, test_size=0.5, stratify=cifar_labels, random_state=42
        )
        train_cifar_tensor, test_cifar_tensor, train_cifar_lbs, test_cifar_lbs = train_test_split(
            cifar_tensor, cifar_labels, test_size=0.2, stratify=cifar_labels, random_state=42
        )
        train_cifar_tensor, val_cifar_tensor, train_cifar_lbs, val_cifar_lbs = train_test_split(
            train_cifar_tensor, train_cifar_lbs, test_size=0.1, stratify=train_cifar_lbs, random_state=42
        )

        torch.save((train_cifar_tensor, train_cifar_lbs), PROCESSED_DIR / "train_cifar.pt")
        print("Train Cifar Tensor Saved.")
        torch.save((val_cifar_tensor, val_cifar_lbs), PROCESSED_DIR / "val_cifar.pt")
        print("Val Cifar Tensor Saved.")
        torch.save((test_cifar_tensor, test_cifar_lbs), PROCESSED_DIR / "test_cifar.pt")
        print("Val Cifar Tensor Saved.")

print("Data processing complete and saved.")


def main():
    #process_data(RAW_DIR, PROCESSED_DIR, 'cifar10')
    process_data(RAW_DIR, PROCESSED_DIR, 'cifar10.1')


if __name__ == "__main__":
    main()
