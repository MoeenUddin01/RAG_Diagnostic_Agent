"""Streamlit interactive UI."""

from __future__ import annotations

import base64
import io

import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from src.model.model import load_checkpoint
from src.utils import get_device

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
)

# Constants
MODEL_PATH = "artifacts/modelpt/best_model.pt"
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Image transforms
TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


@st.cache_resource
def load_model_cached():
    """Load and cache the model.

    Returns:
        Loaded model in evaluation mode.
    """
    device = get_device()
    model, _ = load_checkpoint(
        MODEL_PATH, device, num_classes=len(CLASS_NAMES)
    )
    model.eval()
    return model, device


def predict_image(image: Image.Image, model: torch.nn.Module, device: torch.device):
    """Make prediction on an image.

    Args:
        image: PIL Image to predict.
        model: Trained model.
        device: Compute device.

    Returns:
        Tuple of (predicted_class, confidence, probabilities_dict).
    """
    with torch.no_grad():
        image_tensor = TRANSFORMS(image).unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    probs_dict = {
        name: prob.item() for name, prob in zip(CLASS_NAMES, probabilities[0])
    }

    return predicted_class, confidence.item(), probs_dict


def main() -> None:
    """Run the Streamlit app."""
    st.title("🌿 Plant Disease Classifier")
    st.markdown(
        """
        Upload an image of a plant leaf to detect diseases.
        This model supports Pepper, Potato, and Tomato plants.
        """
    )

    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model_cached()

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, probs = predict_image(
                    image, model, device
                )

            # Display results
            st.subheader("Prediction Results")

            # Main prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Class", predicted_class)
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")

            # Probability distribution
            st.subheader("Class Probabilities")
            prob_df = (
                st.dataframe(
                    [
                        {"Class": name, "Probability": prob}
                        for name, prob in sorted(
                            probs.items(), key=lambda x: x[1], reverse=True
                        )
                    ],
                    hide_index=True,
                )
            )

            # Top 5 classes bar chart
            st.subheader("Top 5 Predictions")
            top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            st.bar_chart({name: prob for name, prob in top_5})

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        """
        This app uses an EfficientNet-B2 model trained on the
        PlantVillage dataset to classify plant diseases.

        **Supported Plants:**
        - Pepper (bell)
        - Potato
        - Tomato
        """
    )


if __name__ == "__main__":
    main()
