"""Dataset utilities using ImageFolder for train/val/test splits."""

from __future__ import annotations

from torchvision.datasets import ImageFolder

from src.data.transforms import (
    get_test_transform,
    get_train_transform,
    get_val_transform,
)


def get_datasets(
    train_dir: str = "dataset/processed/train",
    val_dir: str = "dataset/processed/val",
    test_dir: str = "dataset/processed/test",
) -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """Create ImageFolder datasets for train, validation, and test splits.

    Args:
        train_dir: Path to training data directory.
        val_dir: Path to validation data directory.
        test_dir: Path to test data directory.

    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    test_transform = get_test_transform()

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()

    print(f"Number of training samples: {len(train_ds)}")
    print(f"Number of validation samples: {len(val_ds)}")
    print(f"Number of testing samples: {len(test_ds)}")
    print(f"Class names: {train_ds.classes}")
