"""Pydantic request models for API validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for single image prediction.

    Attributes:
        image_base64: Base64-encoded image string.
    """

    image_base64: str = Field(
        ...,
        description="Base64-encoded image string (PNG/JPG format).",
    )
