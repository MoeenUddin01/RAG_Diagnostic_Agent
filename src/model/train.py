"""Training loop for EfficientNet-B2 plant disease classifier.

Implements a two-phase fine-tuning strategy:

1. **Warm-up** (``freeze_epochs``): Only the classifier head is trained.
2. **Full fine-tune** (remaining epochs): All parameters are unfrozen.

Checkpoints the best model (by validation loss) to ``artifacts/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.model.model import build_model, save_checkpoint, unfreeze_backbone
from src.utils import get_device


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """Execute one epoch of training or validation.

    Args:
        model: The model to train or evaluate.
        loader: DataLoader supplying ``(images, labels)`` batches.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimiser instance; ignored when ``is_train=False``.
        device: Device for tensor placement.
        is_train: When ``True`` gradients are computed and weights updated.

    Returns:
        A two-tuple ``(avg_loss, accuracy)`` over the full epoch.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 15,
    epochs: int = 30,
    freeze_epochs: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    class_names: list[str] | None = None,
    artifacts_dir: Path = Path("artifacts"),
    experiment_name: str = "plant_disease_classifier",
    run_name: str | None = None,
) -> nn.Module:
    """Train EfficientNet-B2 on the plant disease dataset.

    Applies a two-phase strategy:
    - Epochs 0 … ``freeze_epochs - 1``: backbone frozen, only the head trains.
    - Epochs ``freeze_epochs`` … ``epochs - 1``: all parameters unfrozen.

    The model checkpoint with the lowest validation loss is saved to
    ``artifacts/best_model.pt``. A final checkpoint is saved to
    ``artifacts/last_model.pt`` regardless of performance.

    All hyperparameters, metrics, and artifacts are logged to MLflow for
    experiment tracking.

    Args:
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        num_classes: Number of target classes (must match dataset).
        epochs: Total number of training epochs.
        freeze_epochs: Number of initial epochs to keep the backbone frozen.
        lr: Initial learning rate for AdamW.
        weight_decay: L2 regularization coefficient for AdamW.
        patience: Early stopping patience (not currently implemented).
        seed: Random seed for reproducibility (not currently implemented).
        class_names: Optional list of class label names.
        artifacts_dir: Directory where checkpoints are written.
        experiment_name: MLflow experiment name for tracking.
        run_name: Optional MLflow run name for this training run.

    Returns:
        The trained ``nn.Module`` with the best validation-loss weights loaded.

    Raises:
        ValueError: If ``freeze_epochs >= epochs``.
    """
    if freeze_epochs >= epochs:
        raise ValueError(
            f"freeze_epochs ({freeze_epochs}) must be less than epochs ({epochs})."
        )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        device = get_device()
        model = build_model(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=True,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # Calculate batch size from dataloader
        batch_size = len(train_loader.dataset) / len(train_loader)

        # Log hyperparameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "freeze_epochs": freeze_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "seed": seed,
                "batch_size": int(batch_size),
                "num_classes": num_classes,
            }
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        history: list[dict] = []

        print(
            f"Training on {device} for {epochs} epochs "
            f"(backbone frozen for first {freeze_epochs}).\n"
        )

        for epoch in range(epochs):
            epoch_start = time.time()

            # Phase transition: unfreeze backbone and reset optimiser
            if epoch == freeze_epochs:
                print(f"[Epoch {epoch}] Unfreezing backbone — full fine-tune begins.\n")
                unfreeze_backbone(model)
                optimizer = AdamW(
                    model.parameters(), lr=lr / 10, weight_decay=weight_decay
                )
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6
                )

            train_loss, train_acc = _run_epoch(
                model, train_loader, criterion, optimizer, device, is_train=True
            )
            val_loss, val_acc = _run_epoch(
                model, val_loader, criterion, None, device, is_train=False
            )
            scheduler.step()

            elapsed = time.time() - epoch_start
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(record)

            # Log metrics for this epoch
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    artifacts_dir=artifacts_dir,
                    filename="best_model.pt",
                    class_names=class_names,
                )
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        save_checkpoint(
            model,
            optimizer,
            epochs - 1,
            val_loss,
            artifacts_dir=artifacts_dir,
            filename="last_model.pt",
            class_names=class_names,
        )
        print("\nTraining complete.")

        # Log summary metrics
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_val_acc", best_val_acc)

        # Save history to JSON
        history_path = artifacts_dir / "history.json"
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)

        # Log artifacts
        mlflow.log_artifact(str(artifacts_dir / "best_model.pt"))
        mlflow.log_artifact(str(history_path))

    # Return model with best weights
    best_ckpt = artifacts_dir / "best_model.pt"
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    return model
