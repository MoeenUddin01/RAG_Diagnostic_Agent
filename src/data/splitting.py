"""Dataset splitting utilities."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.data.common import build_class_file_map, prepare_output_directory


@dataclass(frozen=True)
class SplitSummary:
    """Summary for the train/val/test split of a single class."""

    class_name: str
    total_count: int
    train_count: int
    val_count: int
    test_count: int


def compute_split_counts(
    total_count: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Compute train, validation, and test counts for a class.

    Args:
        total_count: Total number of files in the class.
        train_ratio: Ratio assigned to the training set.
        val_ratio: Ratio assigned to the validation set.
        test_ratio: Ratio assigned to the test set.

    Returns:
        A tuple of ``(train_count, val_count, test_count)``.

    Raises:
        ValueError: If ratios are invalid.
    """
    if total_count < 1:
        raise ValueError("total_count must be at least 1")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if any(ratio < 0 for ratio in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("Split ratios must be non-negative")

    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count
    return train_count, val_count, test_count


def create_split_dataset(
    source_dir: Path,
    target_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.3,
    test_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> list[SplitSummary]:
    """Split a class-organized dataset into train, validation, and test sets.

    Args:
        source_dir: Root directory of the source class folders.
        target_dir: Output directory that will contain split subdirectories.
        train_ratio: Ratio assigned to the training set.
        val_ratio: Ratio assigned to the validation set.
        test_ratio: Ratio assigned to the test set.
        seed: Random seed for deterministic shuffling.
        overwrite: Whether to replace an existing target directory.

    Returns:
        Per-class summary rows describing the exported split counts.
    """
    class_file_map = build_class_file_map(source_dir)
    prepare_output_directory(target_dir=target_dir, overwrite=overwrite)

    rng = random.Random(seed)
    summaries: list[SplitSummary] = []

    for split_name in ("train", "val", "test"):
        (target_dir / split_name).mkdir(parents=True, exist_ok=True)

    for class_name, files in sorted(class_file_map.items()):
        shuffled_files = list(files)
        rng.shuffle(shuffled_files)
        train_count, val_count, test_count = compute_split_counts(
            total_count=len(shuffled_files),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        split_map = {
            "train": shuffled_files[:train_count],
            "val": shuffled_files[train_count : train_count + val_count],
            "test": shuffled_files[train_count + val_count :],
        }

        for split_name, split_files in split_map.items():
            split_class_dir = target_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for source_file in split_files:
                shutil.copy2(source_file, split_class_dir / source_file.name)

        summaries.append(
            SplitSummary(
                class_name=class_name,
                total_count=len(shuffled_files),
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
            )
        )

    return summaries
