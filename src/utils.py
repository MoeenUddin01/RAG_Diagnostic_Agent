"""Shared project utilities.

Provides device selection and path helpers used across ``src/``.
"""

from __future__ import annotations

from pathlib import Path

import torch


def get_device() -> torch.device:
    """Return the best available compute device.

    Priority: CUDA GPU → Apple MPS → CPU.

    Returns:
        A :class:`torch.device` instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_project_root() -> Path:
    """Return the absolute path to the repository root.

    Resolves upward from this file's location, which is ``src/utils.py``,
    so the root is two levels up.

    Returns:
        A :class:`~pathlib.Path` pointing to the project root directory.
    """
    return Path(__file__).resolve().parent.parent


def get_artifacts_dir() -> Path:
    """Return the ``artifacts/`` directory path, creating it if absent.

    Returns:
        A :class:`~pathlib.Path` to ``<project_root>/artifacts/``.
    """
    artifacts = get_project_root() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts
