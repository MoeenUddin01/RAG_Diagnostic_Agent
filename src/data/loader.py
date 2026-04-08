"""DataLoader utilities for train, validation, and test datasets."""

from __future__ import annotations

from torch.utils.data import DataLoader

from src.data.dataset import get_datasets


def get_dataloaders(
    train_dir: str = "dataset/processed/train",
    val_dir: str = "dataset/processed/val",
    test_dir: str = "dataset/processed/test",
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test datasets.

    Args:
        train_dir: Path to training data directory.
        val_dir: Path to validation data directory.
        test_dir: Path to test data directory.
        batch_size: Number of samples per batch.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset, val_dataset, test_dataset = get_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
