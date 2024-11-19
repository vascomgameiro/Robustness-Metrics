import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants
RESIZE_SHAPE = (64, 64)
FLATTENED_LENGTH = RESIZE_SHAPE[0] * RESIZE_SHAPE[1] * 3

# Directory paths
TRAIN_DIR = Path("data/raw/tiny-imagenet-200/train")
VAL_DIR = Path("data/raw/tiny-imagenet-200/val")
TEST_DIR = Path("data/raw/imagenet-r")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_images(folder_path, label=None):
    images, labels = [], []
    for img_file in folder_path.iterdir():
        if img_file.is_file():
            try:
                with Image.open(img_file).convert("RGB") as img:
                    img = img.resize(RESIZE_SHAPE)
                    img_array = np.array(img).flatten() / 255.0
                    if img_array.size == FLATTENED_LENGTH:
                        images.append(img_array)
                        labels.append(label if label else img_file.stem)
                    else:
                        logging.warning(f"Skipping {img_file.name}: incorrect shape {img_array.shape}")
            except Exception as e:
                logging.warning(f"Error processing {img_file.name}: {e}")
    return images, labels


def get_train_data():
    train_images, train_labels = [], []
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            images, labels = load_images(class_dir / "images", class_dir.name)
            train_images.extend(images)
            train_labels.extend(labels)
    return pd.DataFrame(train_images, columns=[f"pixel_{i}" for i in range(FLATTENED_LENGTH)]).assign(
        label=train_labels
    )


def get_val_data():
    val_annotations = VAL_DIR / "val_annotations.txt"
    val_images, val_labels = [], []
    with val_annotations.open() as f:
        for line in f:
            parts = line.strip().split("\t")
            img_file, label = parts[0], parts[1]
            img_path = VAL_DIR / "images" / img_file
            if img_path.is_file():
                imgs, lbls = load_images(img_path.parent, label)
                val_images.extend(imgs)
                val_labels.extend(lbls)
    return pd.DataFrame(val_images, columns=[f"pixel_{i}" for i in range(FLATTENED_LENGTH)]).assign(label=val_labels)


def get_test_data():
    test_images, test_labels = load_images(TEST_DIR)
    return pd.DataFrame(test_images, columns=[f"pixel_{i}" for i in range(FLATTENED_LENGTH)]).assign(label=test_labels)


def process_data():
    start = time.time()
    logging.info("Processing data...")
    train_df = get_train_data()
    end = time.time()
    logging.info(f"Train data shape: {train_df.shape} ({end - start:.2f} seconds)")
    start = end
    val_df = get_val_data()
    end = time.time()
    logging.info(f"Validation data shape: {val_df.shape} ({end - start:.2f} seconds)")
    start = end
    test_df = get_test_data()
    end = time.time()
    logging.info(f"Test data shape: {test_df.shape} ({end - start:.2f} seconds)")

    common_labels = set(test_df["label"]).intersection(val_df["label"])
    train_df = train_df[train_df["label"].isin(common_labels)]
    val_df = val_df[val_df["label"].isin(common_labels)]
    test_df = test_df[test_df["label"].isin(common_labels)]
    logging.info(f"Common labels: {len(common_labels)}")

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df.pop("label"))
    val_labels = label_encoder.transform(val_df.pop("label"))
    test_labels = label_encoder.transform(test_df.pop("label"))

    def to_tensor(df, labels):
        return (
            torch.tensor(df.values, dtype=torch.float32).view(-1, 3, *RESIZE_SHAPE),
            torch.tensor(labels, dtype=torch.long),
        )

    train_tensor, train_labels_tensor = to_tensor(train_df, train_labels)
    val_tensor, val_labels_tensor = to_tensor(val_df, val_labels)
    test_tensor, test_labels_tensor = to_tensor(test_df, test_labels)

    torch.save((train_tensor, train_labels_tensor), PROCESSED_DIR / "train.pt")
    logging.info("Train data saved.")
    torch.save((val_tensor, val_labels_tensor), PROCESSED_DIR / "val.pt")
    logging.info("Validation data saved.")
    torch.save((test_tensor, test_labels_tensor), PROCESSED_DIR / "test.pt")
    logging.info("Data processing complete and saved.")


def main():
    process_data()

if __name__ == "__main__":
    main()
