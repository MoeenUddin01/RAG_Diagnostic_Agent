"""Prediction route handlers."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import torch
from fastapi import APIRouter, Depends, HTTPException, status
from PIL import Image

from app.dependencies import (
    get_class_names,
    get_device_cached,
    get_model,
    get_transforms,
)
from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from src.model.prediction import predict_single

router = APIRouter(prefix="/v1/prediction", tags=["prediction"])


@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Predict disease class from image",
    status_code=status.HTTP_200_OK,
)
def predict(
    request: PredictionRequest,
    model: torch.nn.Module = Depends(get_model),
    class_names: list[str] = Depends(get_class_names),
    device: torch.device = Depends(get_device_cached),
    transforms = Depends(get_transforms),
) -> PredictionResponse:
    """Predict the plant disease class from an uploaded image.

    Accepts a base64-encoded image and returns the predicted class,
    confidence score, and full probability distribution.

    Args:
        request: Prediction request containing base64-encoded image.
        model: Trained model (injected via dependency).
        class_names: List of class label names (injected via dependency).
        device: Compute device (injected via dependency).
        transforms: Image preprocessing transforms (injected via dependency).

    Returns:
        PredictionResponse containing predicted class, confidence, and
        class probabilities.

    Raises:
        HTTPException: If image decoding or prediction fails.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Save temporarily for prediction function
        temp_path = Path("/tmp/temp_prediction_image.jpg")
        image.save(temp_path)

        # Make prediction
        predicted_idx, confidence, probabilities = predict_single(
            model, temp_path, device, transforms
        )

        # Clean up temp file
        temp_path.unlink()

        # Map to class names
        predicted_class = class_names[predicted_idx]
        class_probs = {
            name: prob.item() for name, prob in zip(class_names, probabilities)
        }

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e
