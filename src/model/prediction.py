"""Model inference utilities.

Provides functions for making predictions on single images or batches
using a trained EfficientNet-B2 model.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.utils import get_device

# Image preprocessing pipeline matching training transforms
INFERENCE_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def predict_single(
    model: nn.Module,
    image_path: Path,
    device: torch.device | None = None,
    transforms: transforms.Compose | None = None,
) -> tuple[int, float, torch.Tensor]:
    """Make a prediction on a single image.

    Args:
        model: Trained EfficientNet-B2 model.
        image_path: Path to the image file.
        device: Device to run inference on. If None, uses best available.
        transforms: Image preprocessing pipeline. If None, uses default.

    Returns:
        A tuple of (predicted_class_index, confidence, probabilities)
        where probabilities is a 1D tensor of class probabilities.

    Raises:
        FileNotFoundError: If image_path does not exist.
        RuntimeError: If image loading or preprocessing fails.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if device is None:
        device = get_device()

    if transforms is None:
        transforms = INFERENCE_TRANSFORMS

    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms(image).unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities.squeeze(0)


def predict_batch(
    model: nn.Module,
    image_paths: list[Path],
    device: torch.device | None = None,
    transforms: transforms.Compose | None = None,
) -> list[tuple[int, float, torch.Tensor]]:
    """Make predictions on a batch of images.

    Args:
        model: Trained EfficientNet-B2 model.
        image_paths: List of paths to image files.
        device: Device to run inference on. If None, uses best available.
        transforms: Image preprocessing pipeline. If None, uses default.

    Returns:
        A list of tuples, each containing (predicted_class_index, confidence,
        probabilities) for the corresponding image.

    Raises:
        FileNotFoundError: If any image_path does not exist.
    """
    results = []
    for image_path in image_paths:
        result = predict_single(model, image_path, device, transforms)
        results.append(result)
    return results

