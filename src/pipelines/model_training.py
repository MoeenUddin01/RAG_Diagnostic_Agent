"""Model training pipeline."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from src.data.loader import get_dataloaders
from src.model.evaluation import evaluate, print_evaluation
from src.model.train import train
from src.utils import get_device


def main() -> None:
    """Run the training pipeline with CLI configuration.

    Parses command-line arguments, validates data directories, loads
    datasets, trains the model with two-phase fine-tuning, and evaluates
    on the validation set.

    Raises:
        SystemExit: If train or validation directories do not exist.
    """
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B2 on plant disease dataset."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("dataset/processed/train"),
        help="Path to training data directory.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("dataset/processed/val"),
        help="Path to validation data directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=5,
        help="Number of initial epochs with backbone frozen.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization coefficient.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--no-weighted-sampler",
        action="store_false",
        dest="use_weighted_sampler",
        help="Disable weighted random sampling for class imbalance.",
    )
    parser.set_defaults(use_weighted_sampler=True)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="plant_disease_classifier",
        help="MLflow experiment name for tracking.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name for this training run.",
    )

    args = parser.parse_args()

    # Validate directories exist
    if not args.train_dir.exists():
        sys.exit(f"Error: Training directory not found: {args.train_dir}")
    if not args.val_dir.exists():
        sys.exit(f"Error: Validation directory not found: {args.val_dir}")

    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        train_dir=str(args.train_dir),
        val_dir=str(args.val_dir),
        batch_size=args.batch_size,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    # Extract class names
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)

    # Print experiment summary
    device = get_device()

    print("=" * 60)
    print("  Training Configuration")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Model: EfficientNet-B2 (num_classes={num_classes})")
    print(f"  Classes: {class_names}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Freeze epochs: {args.freeze_epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    print(f"  Weighted sampler: {args.use_weighted_sampler}")
    print(f"  Artifacts directory: {args.artifacts_dir}")
    print("=" * 60)
    print()

    # Train model
    trained_model = train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        class_names=class_names,
        artifacts_dir=args.artifacts_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )

    # Evaluate on validation set
    results = evaluate(
        model=trained_model,
        loader=val_loader,
        class_names=class_names,
        device=device,
    )
    print_evaluation(results, split_name="val")

    print("\nMLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
