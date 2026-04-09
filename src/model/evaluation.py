"""Model evaluation utilities.

Computes accuracy, per-class precision/recall/F1, and a confusion matrix
over a DataLoader split. Optionally saves a confusion matrix PNG to
``artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import get_device

try:
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str] | None = None,
    device: torch.device | None = None,
) -> dict:
    """Evaluate a model over a DataLoader and return metrics.

    Args:
        model: A trained ``nn.Module`` in eval-compatible state.
        loader: DataLoader for the split to evaluate (val or test).
        class_names: Optional ordered list of class label strings. When
            provided, the classification report uses them as display
            names. Must match the integer label ordering in the dataset.
        device: Device for inference. Defaults to :func:`~src.utils.get_device`.

    Returns:
        A dict with the following keys:

        - ``accuracy`` (float): Overall top-1 accuracy.
        - ``report`` (str): sklearn classification report, or ``""`` if
          sklearn is not installed.
        - ``confusion_matrix`` (list[list[int]]): Raw confusion matrix as
          a nested list, or ``[]`` if sklearn is not installed.
        - ``all_preds`` (list[int]): Flat list of predicted class indices.
        - ``all_labels`` (list[int]): Flat list of true class indices.
    """
    if device is None:
        device = get_device()

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)

    report = ""
    cm: list[list[int]] = []

    if _SKLEARN_AVAILABLE:
        target_names = class_names if class_names else None
        report = classification_report(
            all_labels,
            all_preds,
            target_names=target_names,
            zero_division=0,
        )
        cm = confusion_matrix(all_labels, all_preds).tolist()
    else:
        print(
            "sklearn not installed — skipping classification report "
            "and confusion matrix. Run: pip install scikit-learn"
        )

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def print_evaluation(
    results: dict,
    split_name: str = "test",
) -> None:
    """Pretty-print evaluation results to stdout.

    Args:
        results: The dict returned by :func:`evaluate`.
        split_name: Label for the split (e.g. ``"val"`` or ``"test"``).

    Returns:
        None
    """
    acc = results["accuracy"]
    print(f"\n{'=' * 50}")
    print(f"  Evaluation results — {split_name} split")
    print(f"{'=' * 50}")
    print(f"  Overall accuracy: {acc * 100:.2f}%")
    if results["report"]:
        print("\n  Per-class report:\n")
        print(results["report"])


def save_confusion_matrix(
    results: dict,
    class_names: list[str],
    artifacts_dir: Path = Path("artifacts"),
    filename: str = "confusion_matrix.png",
) -> Path | None:
    """Render and save a confusion matrix heatmap as a PNG.

    Requires ``matplotlib`` and ``seaborn``. If either is missing the
    function logs a warning and returns ``None``.

    Args:
        results: The dict returned by :func:`evaluate`.
        class_names: Ordered list of class label strings.
        artifacts_dir: Directory where the PNG will be saved.
        filename: Output filename.

    Returns:
        The :class:`~pathlib.Path` of the saved image, or ``None`` if
        the required libraries are unavailable.

    Raises:
        OSError: If the file cannot be written to ``artifacts_dir``.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        print(
            "matplotlib/seaborn not installed — skipping confusion matrix plot. "
            "Run: pip install matplotlib seaborn"
        )
        return None

    cm = results["confusion_matrix"]
    if not cm:
        print("No confusion matrix data available.")
        return None

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / filename

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")
    return output_path