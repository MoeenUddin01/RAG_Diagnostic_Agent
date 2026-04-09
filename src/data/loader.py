"""DataLoader utilities for train, validation, and test datasets."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import get_datasets


def get_dataloaders(
    train_dir: str = "dataset/processed/train",
    val_dir: str = "dataset/processed/val",
    test_dir: str = "dataset/processed/test",
    batch_size: int = 32,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation, and test datasets.

    Args:
        train_dir: Path to training data directory.
        val_dir: Path to validation data directory.
        test_dir: Path to test data directory.
        batch_size: Number of samples per batch.
        use_weighted_sampler: If True, use WeightedRandomSampler to handle
            class imbalance by computing inverse-frequency weights from
            train_dataset.targets (e.g., Potato___healthy has only 91 samples
            vs 600 for most classes). Defaults to True.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset, val_dataset, test_dataset = get_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
    )

    if use_weighted_sampler:
        class_counts: dict[int, int] = {}
        for label in train_dataset.targets:
            class_counts[label] = class_counts.get(label, 0) + 1

        class_weight: dict[int, float] = {
            c: 1.0 / count for c, count in class_counts.items()
        }
        sample_weights = [class_weight[label] for label in train_dataset.targets]
        weights_tensor = torch.DoubleTensor(sample_weights)
        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
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
    use_weighted_sampler = True
    train_loader, val_loader, test_loader = get_dataloaders(
        use_weighted_sampler=use_weighted_sampler
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    print(f"Training classes: {train_loader.dataset.classes}")
    print(f"Weighted sampler active: {use_weighted_sampler}")
