# RAG Diagnostic Agent

RAG Diagnostic Agent is a plant disease diagnosis project intended to combine
computer vision with retrieval-augmented generation (RAG). The target workflow
is straightforward: classify a plant image, then retrieve treatment guidance
from a curated knowledge base.

## Current State

This repository is an early scaffold, not a finished application.

- The package structure is in place for training, inference, orchestration,
  FastAPI serving, and a Streamlit UI.
- Most implementation files currently contain module stubs rather than working
  logic.
- The dataset has now been organized into processed `train`, `val`, and `test`
  folders for model development.

If you are contributing to this repo, treat it as a clean foundation for
building the full pipeline rather than a runnable end-to-end product today.

## Planned Architecture

The intended system has two main parts:

1. A vision model that identifies plant disease classes from leaf images.
2. A RAG layer that retrieves supporting treatment or management information
   from a document knowledge base.

Those parts are expected to be connected by the orchestration layer in
`src/orchestrator/`.

## Tech Stack

- Python 3.12+
- PyTorch and TorchVision
- LangChain
- ChromaDB
- Ollama
- FastAPI
- Streamlit
- Pydantic
- Pandas

## Repository Layout

```text
app/
  main.py              FastAPI entry point scaffold
  dependencies.py      Shared FastAPI dependencies scaffold
  middleware.py        Middleware scaffold
  model_loader.py      Serving-time model loading scaffold
  predict.py           API prediction scaffold
  schemas.py           API schema scaffold
  routers/             API router package
  schemas/             Schema package
  templates/           HTML templates
  streamlit.py         Streamlit UI scaffold

src/
  data/                Dataset loading and label encoding scaffolds
  model/               Model loading, training, evaluation, prediction scaffolds
  pipelines/           End-to-end pipeline module scaffolds
  rag/                 Retrieval ingestion scaffold
  vision/              Vision training scaffold
  orchestrator/        Vision + RAG orchestration scaffold
  utils.py             Shared utilities module

dataset/
  raw/                 Raw image dataset directory
  processed/           Split dataset for model training and evaluation
    train/             Training split
    val/               Validation split
    test/              Test split

artifacts/             Saved models and generated outputs
notebook/              Exploratory notebooks
```

## Installation

Using `uv`:

```bash
uv sync
```

Using `pip` and a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Local Development

If you plan to work on the RAG stack locally, start Ollama separately and pull
the model you want to use:

```bash
ollama serve
ollama pull llama3
```

## Dataset Preparation

The balanced image dataset in `dataset/raw/balance dataset` has been copied
into `dataset/processed` using a class-wise split:

- `train`: 60%
- `val`: 30%
- `test`: 10%

Current image counts:

- `train`: 8083
- `val`: 4040
- `test`: 1351

The raw dataset remains unchanged. The processed dataset is organized in the
standard image-classification layout:

```text
dataset/processed/
  train/<class_name>/
  val/<class_name>/
  test/<class_name>/
```

## Entry Points

These modules exist as project entry points, but most are not implemented yet:

```bash
python main.py
python -m src.rag.ingest
python -m src.vision.train
python -m src.orchestrator.main
uvicorn app.main:app --reload
streamlit run app/streamlit.py
```

At the moment, only `python main.py` is expected to produce a visible result,
and that is a simple placeholder message.

## Development Notes

- Keep training and ML pipeline logic inside `src/`.
- Keep API and UI serving concerns inside `app/`.
- Load deployed model artifacts from `artifacts/`.
- Follow PEP 8 and prefer fully typed function signatures.
- Use concise module boundaries so training, serving, and orchestration remain
  separate.

## Suggested Next Steps

- Implement dataset loading and label encoding in `src/data/`.
- Build the training and evaluation loop in `src/model/`.
- Add PDF or document ingestion in `src/rag/ingest.py`.
- Expose inference through `app/main.py` and `app/predict.py`.
- Connect prediction and retrieval in `src/orchestrator/main.py`.

## Project Metadata

- Package name: `rag-diagnostic-agent`
- Version: `0.1.0`
- Python requirement: `>=3.12`
