"""Dataset balancing utilities."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.data.common import build_class_file_map, prepare_output_directory


@dataclass(frozen=True)
class ClassSelectionSummary:
    """Summary for the balanced dataset export of a single class."""

    class_name: str
    source_count: int
    selected_count: int

    @property
    def removed_count(self) -> int:
        """Return the number of images excluded from the balanced export."""
        return self.source_count - self.selected_count


def select_balanced_files(
    class_file_map: dict[str, list[Path]],
    max_images_per_class: int,
    seed: int,
) -> tuple[dict[str, list[Path]], list[ClassSelectionSummary]]:
    """Select up to a fixed number of images per class.

    Args:
        class_file_map: Mapping of class names to source image files.
        max_images_per_class: Maximum number of images to keep per class.
        seed: Random seed for deterministic sampling.

    Returns:
        The selected files by class, plus summary rows for each class.

    Raises:
        ValueError: If ``max_images_per_class`` is less than 1.
    """
    if max_images_per_class < 1:
        raise ValueError("max_images_per_class must be at least 1")

    rng = random.Random(seed)
    selected_files_by_class: dict[str, list[Path]] = {}
    summaries: list[ClassSelectionSummary] = []

    for class_name, files in sorted(class_file_map.items()):
        if len(files) <= max_images_per_class:
            selected_files = list(files)
        else:
            selected_files = sorted(
                rng.sample(files, max_images_per_class),
                key=lambda path: path.name,
            )

        selected_files_by_class[class_name] = selected_files
        summaries.append(
            ClassSelectionSummary(
                class_name=class_name,
                source_count=len(files),
                selected_count=len(selected_files),
            )
        )

    return selected_files_by_class, summaries


def copy_selected_files(
    selected_files_by_class: dict[str, list[Path]],
    target_dir: Path,
) -> None:
    """Copy selected images into class subdirectories.

    Args:
        selected_files_by_class: Mapping of class names to selected image files.
        target_dir: Destination dataset directory.

    Raises:
        OSError: If a file copy fails.
    """
    for class_name, files in sorted(selected_files_by_class.items()):
        class_target_dir = target_dir / class_name
        class_target_dir.mkdir(parents=True, exist_ok=True)
        for source_file in files:
            shutil.copy2(source_file, class_target_dir / source_file.name)


def create_balanced_dataset(
    source_dir: Path,
    target_dir: Path,
    max_images_per_class: int = 1000,
    seed: int = 42,
    overwrite: bool = False,
) -> list[ClassSelectionSummary]:
    """Create a balanced dataset by capping each class to a max count.

    Args:
        source_dir: Root directory of the source class folders.
        target_dir: Output directory for the balanced dataset.
        max_images_per_class: Maximum images retained per class.
        seed: Random seed for deterministic sampling.
        overwrite: Whether to replace an existing target directory.

    Returns:
        Per-class summary rows describing the balanced export.
    """
    class_file_map = build_class_file_map(source_dir)
    selected_files_by_class, summaries = select_balanced_files(
        class_file_map=class_file_map,
        max_images_per_class=max_images_per_class,
        seed=seed,
    )
    prepare_output_directory(target_dir=target_dir, overwrite=overwrite)
    copy_selected_files(
        selected_files_by_class=selected_files_by_class,
        target_dir=target_dir,
    )
    return summaries
