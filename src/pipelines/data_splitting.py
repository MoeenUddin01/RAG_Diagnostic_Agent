"""Dataset splitting pipeline."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.data.splitting import create_split_dataset
from src.pipelines.data_preprocessing import format_summary_table


DEFAULT_SOURCE_DIR = Path("dataset/raw/balance dataset")
DEFAULT_TARGET_DIR = Path("dataset/processed")
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.3
DEFAULT_TEST_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42


def build_parser() -> ArgumentParser:
    """Build the command-line parser for dataset splitting.

    Returns:
        Configured argument parser for the splitting pipeline.
    """
    parser = ArgumentParser(
        description="Split a class-organized image dataset into train/val/test."
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
        help="Output directory for the split dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of images assigned to training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of images assigned to validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Fraction of images assigned to testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for deterministic shuffling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the target directory if it already exists.",
    )
    return parser


def run(args: Namespace) -> int:
    """Run the dataset splitting pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code.
    """
    summaries = create_split_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    rows = [
        (
            summary.class_name,
            summary.total_count,
            summary.train_count,
            summary.val_count,
            summary.test_count,
        )
        for summary in summaries
    ]
    print(f"Split dataset created at: {args.target_dir}")
    print(
        format_summary_table(
            rows=rows,
            headers=("class_name", "total", "train", "val", "test"),
        )
    )
    print(f"Total classes: {len(summaries)}")
    print(f"Total train images: {sum(row[2] for row in rows):,}")
    print(f"Total val images: {sum(row[3] for row in rows):,}")
    print(f"Total test images: {sum(row[4] for row in rows):,}")
    return 0


def main() -> int:
    """Run the splitting pipeline from the command line.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
