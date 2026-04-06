"""Shared helpers for dataset preparation."""

from __future__ import annotations

import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(class_dir: Path) -> list[Path]:
    """Return sorted image files inside a class directory.

    Args:
        class_dir: Directory containing images for a single class.

    Returns:
        A sorted list of image file paths.

    Raises:
        FileNotFoundError: If ``class_dir`` does not exist.
        NotADirectoryError: If ``class_dir`` is not a directory.
    """
    if not class_dir.exists():
        raise FileNotFoundError(f"Class directory does not exist: {class_dir}")
    if not class_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {class_dir}")

    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_class_file_map(dataset_dir: Path) -> dict[str, list[Path]]:
    """Build a mapping of class names to image file paths.

    Args:
        dataset_dir: Root directory whose child directories represent classes.

    Returns:
        A mapping of class directory names to sorted image file lists.

    Raises:
        FileNotFoundError: If ``dataset_dir`` does not exist.
        NotADirectoryError: If ``dataset_dir`` is not a directory.
        ValueError: If no class directories are found.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {dataset_dir}")

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in: {dataset_dir}")

    return {class_dir.name: list_image_files(class_dir) for class_dir in class_dirs}


def prepare_output_directory(target_dir: Path, overwrite: bool) -> None:
    """Prepare an output directory for export.

    Args:
        target_dir: Directory that will receive exported files.
        overwrite: Whether an existing directory may be removed.

    Raises:
        FileExistsError: If ``target_dir`` exists and ``overwrite`` is false.
    """
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target_dir} already exists. Use overwrite=True to replace it."
            )
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)


def summarize_dataset(dataset_dir: Path) -> dict[str, int]:
    """Summarize a class-organized dataset by class name.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        A mapping of class names to image counts.
    """
    class_file_map = build_class_file_map(dataset_dir)
    return {
        class_name: len(files)
        for class_name, files in sorted(class_file_map.items())
    }
