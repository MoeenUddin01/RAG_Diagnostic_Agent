"""Prediction route handlers."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from fastapi import APIRouter, Depends, HTTPException, status
from PIL import Image

from app.dependencies import (
    get_class_names,
    get_device_cached,
    get_model,
    get_transforms,
    get_vector_store,
)
from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from src.model.prediction import predict_single
from src.orchestrator.main import generate_diagnostic_report

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma

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
    vector_store: "Chroma" = Depends(get_vector_store),
) -> PredictionResponse:
    """Predict the plant disease class from an uploaded image.

    Accepts a base64-encoded image and returns the predicted class,
    confidence score, full probability distribution, and AI-generated
    treatment recommendations from RAG + Groq LLM.

    Args:
        request: Prediction request containing base64-encoded image.
        model: Trained model (injected via dependency).
        class_names: List of class label names (injected via dependency).
        device: Compute device (injected via dependency).
        transforms: Image preprocessing transforms (injected via dependency).
        vector_store: ChromaDB vector store for RAG retrieval (injected via dependency).

    Returns:
        PredictionResponse containing predicted class, confidence,
        class probabilities, and treatment recommendations.

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

        # Retrieve relevant documents from knowledge base
        try:
            search_query = f"{predicted_class} treatment"
            retrieved_docs = vector_store.similarity_search(search_query, k=3)

            # Generate treatment recommendations using Groq LLM
            treatment_report = generate_diagnostic_report(
                class_name=predicted_class,
                confidence=confidence,
                retrieved_docs=retrieved_docs,
                model_name="llama-3.1-8b-instant",  # Use fast model for API speed
            )
        except Exception as rag_error:
            # If RAG fails, still return prediction but with None treatment
            treatment_report = None

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
            treatment_recommendations=treatment_report,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e
