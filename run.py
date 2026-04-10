#!/usr/bin/env python3
"""Run pipeline from YAML configuration file.

Usage:
    python run.py config.yaml          # Run training with production config
    python run.py config.dev.yaml      # Run training with dev/smoke-test config
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import sys
from pathlib import Path

import yaml

from src.data.loader import get_dataloaders
from src.model.evaluation import evaluate, print_evaluation
from src.model.train import train
from src.utils import get_device


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    """Run training pipeline using YAML configuration.

    Parses command-line arguments, loads configuration, validates directories,
    loads datasets, trains the model, and evaluates on validation set.

    Raises:
        FileNotFoundError: If config file or data directories do not exist.
        SystemExit: If validation fails.
    """
    parser = argparse.ArgumentParser(
        description="Run pipeline from YAML configuration file."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Pipeline mode: train or eval (default: train).",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configuration sections
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    artifacts_cfg = config["artifacts"]
    mlflow_cfg = config["mlflow"]
    model_cfg = config["model"]

    # Validate directories exist
    train_dir = Path(dataset_cfg["train_dir"])
    val_dir = Path(dataset_cfg["val_dir"])

    if not train_dir.exists():
        sys.exit(f"Error: Training directory not found: {train_dir}")
    if not val_dir.exists():
        sys.exit(f"Error: Validation directory not found: {val_dir}")

    # Create artifacts directory
    artifacts_dir = Path(artifacts_cfg["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        batch_size=training_cfg["batch_size"],
        use_weighted_sampler=dataset_cfg["use_weighted_sampler"],
    )

    # Extract class names
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)

    # Print experiment summary
    device = get_device()

    print("=" * 60)
    print("  Training Configuration")
    print("=" * 60)
    print(f"  Config file: {args.config}")
    print(f"  Device: {device}")
    print(f"  Model: {model_cfg['architecture']} (num_classes={num_classes})")
    print(f"  Classes: {class_names}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {training_cfg['batch_size']}")
    print(f"  Epochs: {training_cfg['epochs']}")
    print(f"  Freeze epochs: {training_cfg['freeze_epochs']}")
    print(f"  Learning rate: {training_cfg['lr']}")
    print(f"  Weight decay: {training_cfg['weight_decay']}")
    print(f"  Patience: {training_cfg['patience']}")
    print(f"  Seed: {training_cfg['seed']}")
    print(f"  Max batches per epoch: {training_cfg.get('max_batches', 'None (all)')}")
    print(f"  Weighted sampler: {dataset_cfg['use_weighted_sampler']}")
    print(f"  Artifacts directory: {artifacts_dir}")
    print(f"  MLflow experiment: {mlflow_cfg['experiment_name']}")
    print(f"  MLflow run name: {mlflow_cfg.get('run_name', 'auto')}")
    print("=" * 60)
    print()

    # Train model
    trained_model = train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        epochs=training_cfg["epochs"],
        freeze_epochs=training_cfg["freeze_epochs"],
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        patience=training_cfg["patience"],
        seed=training_cfg["seed"],
        class_names=class_names,
        artifacts_dir=artifacts_dir,
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=mlflow_cfg.get("run_name"),
        max_batches=training_cfg.get("max_batches"),
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
