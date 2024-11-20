import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TINY_DIR = RAW_DIR / "tiny-imagenet-200"
R_DIR = RAW_DIR / "imagenet-r"


def load_images(folder_path, label=None):
    images, labels = [], []
    for img_file in folder_path.iterdir():
        if img_file.is_file():
            try:
                with Image.open(img_file).convert("RGB") as img:
                    img = img.resize((64, 64))
                    img_array = np.array(img).flatten() / 255.0
                    if img_array.size == 3 * 64 * 64:
                        images.append(img_array)
                        labels.append(label if label else img_file.stem)
                    else:
                        print(f"Skipping {img_file.name}: incorrect shape {img_array.shape}")
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
    return images, labels


def get_tiny_data(tiny_dir):
    train_images, train_labels = [], []
    train_dir = tiny_dir / "train"
    val_dir = tiny_dir / "val"
    val_annotations = val_dir / "val_annotations.txt"

    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            print("Tiny Train: ", class_dir.name)
            images, labels = load_images(class_dir / "images", class_dir.name)
            train_images.extend(images)
            train_labels.extend(labels)

    val_images, val_labels = [], []
    with val_annotations.open() as f:
        for line in f:
            try:
                parts = line.strip().split("\t")
                img_file, label = parts[0], parts[1]
                img_path = val_dir / "images" / img_file
                print("Tiny Val: ", img_file)
                img = Image.open(img_path).convert("RGB")
                img = img.resize((64, 64))
                img_array = np.array(img).flatten() / 255.0
                val_images.append(img_array)
                val_labels.append(label)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

    tiny_images = np.array(train_images + val_images)
    tiny_labels = np.array(train_labels + val_labels)
    return tiny_images, tiny_labels


def get_r_data(r_dir):
    r_images, r_labels = [], []

    for class_dir in r_dir.iterdir():
        if class_dir.is_dir():
            print("R: ", class_dir.name)
            images, labels = load_images(class_dir, class_dir.name)
            r_images.extend(images)
            r_labels.extend(labels)

    return np.array(r_images), np.array(r_labels)


def to_tensor(images, labels):
    return (
        torch.tensor(images, dtype=torch.float32).view(-1, 3, 64, 64),
        torch.tensor(labels, dtype=torch.long),
    )


def process_data(TINY_DIR, R_DIR, PROCESSED_DIR):
    start = time.time()
    print("Processing data...")
    tiny_images, tiny_labels = get_tiny_data(TINY_DIR)
    end = time.time()
    print(f"Tiny data shape: {tiny_images.shape} ({end - start:.2f} seconds)")
    start = end
    r_images, r_labels = get_r_data(R_DIR)
    end = time.time()
    print(f"R data shape: {r_images.shape} ({end - start:.2f} seconds)")

    common_labels = list(set(r_labels).intersection(tiny_labels))

    tiny_mask = np.isin(tiny_labels, common_labels)
    tiny_images = tiny_images[tiny_mask]
    tiny_labels = tiny_labels[tiny_mask]

    r_mask = np.isin(r_labels, common_labels)
    r_images = r_images[r_mask]
    r_labels = r_labels[r_mask]

    print(f"Common labels: {len(common_labels)}")
    print(f"Filtered Tiny data shape: {tiny_images.shape}")
    print(f"Filtered R data shape: {r_images.shape}")

    label_encoder = LabelEncoder()
    tiny_labels = label_encoder.fit_transform(tiny_labels)
    r_labels = label_encoder.transform(r_labels)

    tiny_tensor, tiny_labels_tensor = to_tensor(tiny_images, tiny_labels)
    r_tensor, r_labels_tensor = to_tensor(r_images, r_labels)

    torch.save((tiny_tensor, tiny_labels_tensor), PROCESSED_DIR / "tiny.pt")
    print("Tiny Tensor Saved.")
    torch.save((r_tensor, r_labels_tensor), PROCESSED_DIR / "r.pt")
    print("R Tensor Saved.")

    train_tiny_tensor, test_tiny_tensor, train_tiny_lbs, test_tiny_lbs = train_test_split(
        tiny_tensor, tiny_labels, test_size=0.2, stratify=tiny_labels, random_state=42
    )
    train_tiny_tensor, val_tiny_tensor, train_tiny_lbs, val_tiny_lbs = train_test_split(
        train_tiny_tensor, train_tiny_lbs, test_size=0.1, stratify=train_tiny_lbs, random_state=42
    )
    torch.save((train_tiny_tensor, train_tiny_lbs), PROCESSED_DIR / "train_tiny.pt")
    print("Train Tiny Tensor Saved.")
    torch.save((val_tiny_tensor, val_tiny_lbs), PROCESSED_DIR / "val_tiny.pt")
    print("Val Tiny Tensor Saved.")
    torch.save((test_tiny_tensor, test_tiny_lbs), PROCESSED_DIR / "test_tiny.pt")
    print("Val Tiny Tensor Saved.")

    print("Data processing complete and saved.")


def main():
    process_data(TINY_DIR, R_DIR, PROCESSED_DIR)


if __name__ == "__main__":
    main()
