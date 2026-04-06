# RAG Diagnostic Agent

Multi-modal AI project for plant disease diagnosis that combines a vision
pipeline with retrieval-augmented generation (RAG). The intended flow is:
an image is classified for likely disease symptoms, then the system retrieves
relevant treatment guidance from a knowledge base.

## Current Status

This repository currently provides the project structure and module scaffolding
for the full system. Most runtime modules are placeholders with docstrings, so
the codebase is best understood as a clean starting point for implementation
rather than a finished application.

## Tech Stack

- Python 3.12+
- PyTorch and TorchVision for the vision pipeline
- LangChain, ChromaDB, and Ollama for RAG components
- FastAPI for API serving
- Streamlit for an interactive UI
- Pydantic for request and response schemas

## Repository Structure

```text
app/
  main.py              FastAPI application entry point
  dependencies.py      Shared FastAPI dependencies
  middleware.py        Middleware and exception handling
  model_loader.py      Serving-time model loading
  predict.py           Inference entry point for the API layer
  schemas.py           API schemas
  routers/             API router package
  schemas/             Schema package
  templates/           HTML templates
  streamlit.py         Streamlit user interface

src/
  data/                Dataset loading and label encoding
  model/               Model loading, training, evaluation, prediction
  pipelines/           End-to-end pipeline scripts
  rag/                 Retrieval ingestion components
  vision/              Vision training entry points
  orchestrator/        Vision + RAG orchestration
  utils.py             Shared utilities

dataset/
  raw/                 Plant image dataset

artifacts/
  Model and pipeline outputs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using the local RAG stack with Ollama:

```bash
ollama serve
ollama pull llama3
```

## Common Commands

The repository guidance currently points to these execution commands:

```bash
python -m src.rag.ingest
python -m src.vision.train
python -m src.orchestrator.main
streamlit run app/streamlit.py
uvicorn app.main:app --reload
```

Some of these entry points are scaffolded and may require implementation before
they run successfully.

## Development Notes

- Follow PEP 8 with a maximum line length of 88 characters.
- Use type annotations on all function signatures.
- Add Google-style docstrings to public classes, methods, and functions.
- Keep training logic in `src/` and serving logic in `app/`.
- Load serving-time models from `artifacts/`, not from training code paths.

## Intended Workflow

1. Train or load the vision model.
2. Ingest the knowledge base for retrieval.
3. Run the orchestrator or API layer to combine prediction with retrieved
   treatment guidance.

## Notes

- The dataset included in `dataset/raw/` appears to be PlantVillage-style
  imagery for plant disease classification.
- The project metadata is defined in `pyproject.toml` under the package name
  `rag-diagnostic-agent`.
