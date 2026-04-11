"""Shared dependency injection providers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from fastapi import Depends
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.model.model import load_checkpoint
from src.model.prediction import INFERENCE_TRANSFORMS
from src.utils import get_device, VECTOR_DB_PATH

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma as ChromaType

MODEL_PATH = Path("artifacts/modelpt/best_model.pt")
CONFIG_PATH = Path("config.yaml")


@lru_cache(maxsize=1)
def get_config() -> dict:
    """Load and cache the configuration from config.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config.yaml does not exist.
    """
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_class_names() -> list[str]:
    """Load and cache class names from configuration.

    Returns:
        List of class label names.
    """
    config = get_config()
    return config.get("classes", [])


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    """Load and cache the trained model.

    The model is loaded once at startup and reused for all requests.

    Returns:
        Trained EfficientNet-B2 model in evaluation mode.

    Raises:
        FileNotFoundError: If the model checkpoint does not exist.
    """
    device = get_device()
    config = get_config()
    num_classes = config.get("dataset", {}).get("num_classes", 15)

    model, _ = load_checkpoint(MODEL_PATH, device, num_classes=num_classes)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_device_cached() -> torch.device:
    """Get and cache the compute device.

    Returns:
        The best available torch device.
    """
    return get_device()


def get_transforms():
    """Get the image preprocessing transforms.

    Returns:
        torchvision transforms composition for inference.
    """
    return INFERENCE_TRANSFORMS


@lru_cache(maxsize=1)
def get_vector_store() -> "ChromaType":
    """Load and cache the ChromaDB vector store for RAG retrieval.

    The vector store is loaded once at startup and reused for all requests.

    Returns:
        Chroma vector store instance with agricultural knowledge base.

    Raises:
        ValueError: If vector store is not initialized (run `python -m src.rag.ingest`).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not VECTOR_DB_PATH.exists():
        raise ValueError(
            f"Vector store not found at {VECTOR_DB_PATH}. "
            "Please run: python -m src.rag.ingest"
        )

    return Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings,
    )
