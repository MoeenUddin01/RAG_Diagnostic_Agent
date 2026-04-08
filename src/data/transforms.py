"""Image transforms for train, validation, and test datasets."""

from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)


def get_train_transform() -> transforms.Compose:
    """Return transform pipeline for training data with augmentation.

    Returns:
        Composed transform with resize, horizontal flip, tensor conversion,
        and ImageNet normalization.
    """
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """Return transform pipeline for validation data.

    Returns:
        Composed transform with resize, tensor conversion, and
        ImageNet normalization. No random augmentation.
    """
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_test_transform() -> transforms.Compose:
    """Return transform pipeline for test data.

    Returns:
        Composed transform with resize, tensor conversion, and
        ImageNet normalization. No random augmentation.
    """
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
