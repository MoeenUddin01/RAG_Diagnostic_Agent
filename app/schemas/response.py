"""Pydantic response models for API responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response model for prediction results.

    Attributes:
        predicted_class: The predicted disease/healthy class label.
        confidence: Model confidence score (0-1).
        class_probabilities: Dictionary mapping class names to probabilities.
        treatment_recommendations: AI-generated treatment report from RAG + LLM.
    """

    predicted_class: str = Field(
        ...,
        description="The predicted disease/healthy class label.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0 and 1.",
    )
    class_probabilities: dict[str, float] = Field(
        ...,
        description="Dictionary mapping class names to their probabilities.",
    )
    treatment_recommendations: str | None = Field(
        default=None,
        description="AI-generated treatment report from RAG + LLM (None if RAG unavailable).",
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: Service health status.
        model_loaded: Whether the model is loaded.
        device: Compute device being used.
    """

    status: str = Field(..., description="Service health status.")
    model_loaded: bool = Field(..., description="Whether the model is loaded.")
    device: str = Field(..., description="Compute device being used.")
