"""Data preprocessing pipeline."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from src.data.balancing import create_balanced_dataset


DEFAULT_SOURCE_DIR = Path("dataset/raw/PlantVillage")
DEFAULT_TARGET_DIR = Path("dataset/raw/balance dataset")
DEFAULT_MAX_IMAGES_PER_CLASS = 1000
DEFAULT_RANDOM_SEED = 42


def build_parser() -> ArgumentParser:
    """Build the command-line parser for dataset balancing.

    Returns:
        Configured argument parser for the balancing pipeline.
    """
    parser = ArgumentParser(
        description="Create a balanced image dataset with a per-class cap."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source dataset directory with class subfolders.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Output directory for the balanced dataset.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_CLASS,
        help="Maximum number of images to keep per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for deterministic sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the target directory if it already exists.",
    )
    return parser


def format_summary_table(
    rows: list[tuple[Any, ...]],
    headers: tuple[str, ...],
) -> str:
    """Format summary rows as a plain-text table.

    Args:
        rows: Summary rows to print.
        headers: Table header labels.

    Returns:
        A plain-text table.
    """
    widths = [
        max(len(str(value)) for value in [header, *[row[index] for row in rows]])
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(
            str(header).ljust(widths[index]) for index, header in enumerate(headers)
        ),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    lines.extend(
        "  ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(row)
        )
        for row in rows
    )
    return "\n".join(lines)


def run(args: Namespace) -> int:
    """Run the dataset balancing pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    summaries = create_balanced_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    rows = [
        (
            summary.class_name,
            summary.source_count,
            summary.selected_count,
            summary.removed_count,
        )
        for summary in summaries
    ]
    print(f"Balanced dataset created at: {args.target_dir}")
    print(
        format_summary_table(
            rows=rows,
            headers=("class_name", "source", "selected", "removed"),
        )
    )
    print(f"Total classes: {len(summaries)}")
    print(f"Total images in balanced dataset: {sum(row[2] for row in rows):,}")
    return 0


def main() -> int:
    """Run the balancing pipeline from the command line.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
