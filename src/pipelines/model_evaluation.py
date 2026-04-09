"""Model evaluation pipeline."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

from src.data.loader import get_test_dataloader
from src.model.evaluation import evaluate, print_evaluation, save_confusion_matrix
from src.model.model import load_checkpoint
from src.utils import get_device


def main() -> None:
    """Run the evaluation pipeline with CLI configuration.

    Loads a trained model checkpoint, evaluates it on the test split,
    prints metrics, and optionally saves a confusion matrix.

    Raises:
        SystemExit: If test directory or checkpoint file does not exist.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientNet-B2 on test dataset."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("dataset/processed/test"),
        help="Path to test data directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/best_model.pt"),
        help="Path to model checkpoint file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saving evaluation artifacts.",
    )
    parser.add_argument(
        "--save-cm",
        action="store_true",
        help="Save confusion matrix as PNG.",
    )

    args = parser.parse_args()

    # Validate test directory and checkpoint exist
    if not args.test_dir.exists():
        sys.exit(f"Error: Test directory not found: {args.test_dir}")
    if not args.checkpoint.exists():
        sys.exit(f"Error: Checkpoint file not found: {args.checkpoint}")

    # Load checkpoint
    device = get_device()
    model, checkpoint_dict = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Extract checkpoint info
    epoch = checkpoint_dict.get("epoch", "unknown")
    val_loss = checkpoint_dict.get("val_loss", "unknown")
    class_names = checkpoint_dict.get("class_names")

    if class_names is None:
        print(
            "ERROR: Checkpoint does not contain class_names. "
            "Please retrain the model using the updated model_training.py pipeline."
        )
        sys.exit(1)

    # Print checkpoint info
    print("=" * 60)
    print("  Checkpoint Information")
    print("=" * 60)
    print(f"  Checkpoint path: {args.checkpoint}")
    print(f"  Saved at epoch: {epoch}")
    print(f"  Validation loss: {val_loss}")
    print("=" * 60)
    print()

    # Load test data
    test_loader = get_test_dataloader(
        test_dir=str(args.test_dir),
        batch_size=args.batch_size,
    )

    # Print evaluation summary
    print("=" * 60)
    print("  Evaluation Configuration")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Classes: {class_names}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 60)
    print()

    # Evaluate on test set
    results = evaluate(
        model=model,
        loader=test_loader,
        class_names=class_names,
        device=device,
    )
    print_evaluation(results, split_name="test")

    # Save confusion matrix if requested
    if args.save_cm:
        cm_path = save_confusion_matrix(
            results=results,
            class_names=class_names,
            artifacts_dir=args.artifacts_dir,
            filename="confusion_matrix_test.png",
        )
        if cm_path is not None:
            print(f"\nConfusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
