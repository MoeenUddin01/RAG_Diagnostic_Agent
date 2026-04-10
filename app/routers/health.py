"""Health check route handlers."""

from __future__ import annotations

import torch
from fastapi import APIRouter, Depends

from app.dependencies import get_device_cached, get_model
from app.schemas.response import HealthResponse

router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health check endpoint",
    status_code=200,
)
def health_check(
    model: torch.nn.Module = Depends(get_model),
    device: torch.device = Depends(get_device_cached),
) -> HealthResponse:
    """Check if the API and model are running correctly.

    Returns:
        HealthResponse containing service status and model information.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
    )
