"""Model loading, instantiation, and checkpoint management.

Handles building EfficientNet-B2 with a custom classification head,
saving checkpoints to ``artifacts/``, and restoring them for inference
or resumed training.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B2_Weights

NUM_CLASSES: int = 15
_CLASSIFIER_IN_FEATURES: int = 1408  # EfficientNet-B2 penultimate dim


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Instantiate EfficientNet-B2 with a custom classification head.

    The backbone is loaded with ImageNet-1k weights. The original
    classifier is replaced by a single ``Linear`` layer projecting from
    1408 to ``num_classes``.

    Args:
        num_classes: Number of output disease/healthy classes.
        pretrained: When ``True`` load ImageNet weights; otherwise use
            random initialisation.
        freeze_backbone: When ``True`` freeze all parameters except the
            new classifier head (useful for warm-up epochs).

    Returns:
        A ``torch.nn.Module`` ready for training or inference.
    """
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model: nn.Module = models.efficientnet_b2(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(_CLASSIFIER_IN_FEATURES, num_classes),
    )

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all model parameters in-place.

    Call this after warm-up epochs to enable end-to-end fine-tuning.

    Args:
        model: The EfficientNet-B2 model returned by :func:`build_model`.

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    artifacts_dir: Path = Path("artifacts"),
    filename: str = "checkpoint.pt",
) -> Path:
    """Persist model and optimiser state to disk.

    Args:
        model: The model whose ``state_dict`` will be saved.
        optimizer: The optimiser whose ``state_dict`` will be saved.
        epoch: Current epoch index (0-based).
        val_loss: Validation loss at this checkpoint.
        artifacts_dir: Directory where the file will be written.
        filename: Name of the ``.pt`` file.

    Returns:
        The full :class:`~pathlib.Path` of the saved checkpoint file.

    Raises:
        OSError: If the directory cannot be created or the file cannot
            be written.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> tuple[nn.Module, dict]:
    """Restore a model from a saved checkpoint.

    The backbone is always unfrozen after loading so the model is
    immediately usable for inference or continued training.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device: The :class:`torch.device` to map tensors onto.
        num_classes: Must match the value used when saving the checkpoint.

    Returns:
        A two-tuple of ``(model, checkpoint_dict)`` where
        ``checkpoint_dict`` contains ``epoch``, ``val_loss``, and the
        raw state dicts for downstream use (e.g. restoring an optimiser).

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint: dict = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint