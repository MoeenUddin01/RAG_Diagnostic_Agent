---
title: PlantDoc - AI Plant Diagnostics
emoji: 🌿
colorFrom: green
colorTo: lime
sdk: docker
pinned: false
license: mit
---

# PlantDoc - AI Plant Diagnostics

An AI-powered plant disease diagnosis system using EfficientNet-B2 deep learning model combined with Retrieval-Augmented Generation (RAG) for accurate disease identification and treatment recommendations.

## 🌟 Features

- **Deep Learning Model**: EfficientNet-B2 CNN trained on 13,474 plant disease images
- **15 Disease Classes**: Comprehensive coverage across Pepper, Potato, and Tomato crops
- **RAG-Powered Insights**: Treatment recommendations using LangChain + ChromaDB
- **FastAPI Backend**: High-performance REST API
- **Modern UI**: Clean, responsive web interface

## 🚀 Deployment Instructions

### Option 1: Deploy to Hugging Face Spaces (Docker)

1. Create a new Space on Hugging Face with the "Docker" SDK
2. Push this repository to your Space
3. The Dockerfile will automatically build and deploy the application

### Option 2: Deploy to Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel`
3. Follow the prompts to deploy

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 7860
```

## 📋 Requirements

- Python 3.12+
- PyTorch 2.2.0+
- FastAPI 0.135.3+
- LangChain + ChromaDB
- Model checkpoint at `artifacts/modelpt/best_model.pt`
- Vector database at `data/vector_db/`

## 🔧 Configuration

Ensure the following files are present before deployment:
- `config.yaml` - Model and dataset configuration
- `artifacts/modelpt/best_model.pt` - Trained model checkpoint
- `data/vector_db/` - ChromaDB vector store (run `python -m src.rag.ingest` to create)

## 📊 Model Performance

- **Architecture**: EfficientNet-B2
- **Training Data**: PlantVillage dataset (13,474 images)
- **Classes**: 15 disease types across 3 crops
- **Accuracy**: 95%+ on test set

## 🤝 Contributing

This is an open-source project. Contributions are welcome!

## 📄 License

MIT License
