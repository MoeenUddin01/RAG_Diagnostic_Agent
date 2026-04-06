# RAG Diagnostic Agent

RAG Diagnostic Agent is a plant disease diagnosis project intended to combine
computer vision with retrieval-augmented generation (RAG). The target workflow
is straightforward: classify a plant image, then retrieve treatment guidance
from a curated knowledge base.

## Current State

This repository is an early scaffold, not a finished application.

- The package structure is in place for training, inference, orchestration,
  FastAPI serving, and a Streamlit UI.
- Dataset preprocessing utilities and runnable data pipeline scripts are now
  implemented.
- Vision, RAG, API, and UI modules are still mostly scaffolds.

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
  data/                Dataset balancing, splitting, loading, and label encoding
  model/               Model loading, training, evaluation, prediction scaffolds
  pipelines/           Runnable dataset preparation and ML pipeline entry points
  rag/                 Retrieval ingestion scaffold
  vision/              Vision training scaffold
  orchestrator/        Vision + RAG orchestration scaffold
  utils.py             Shared utilities module

dataset/
  raw/                 Raw image dataset directory
    PlantVillage/      Original class-organized dataset
    balance dataset/   Balanced export capped per class
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

The repository now separates dataset preparation into dedicated modules:

- `src/data/balancing.py` for balancing the raw dataset
- `src/data/splitting.py` for train/val/test splitting
- `src/data/common.py` for shared dataset filesystem helpers

Runnable pipeline scripts remain in `src/pipelines/`.

### 1. Balance the raw dataset

This matches the logic from
`notebook/plantvillage_balance_dataset.ipynb`: it reads
`dataset/raw/PlantVillage`, caps each class at 1000 images, and writes the
result to `dataset/raw/balance dataset`.

```bash
python -m src.pipelines.data_preprocessing
```

Optional flags:

- `--source-dir`
- `--target-dir`
- `--max-images-per-class`
- `--seed`
- `--overwrite`

### 2. Split the balanced dataset

The balanced dataset in `dataset/raw/balance dataset` has been copied into
`dataset/processed` using a class-wise split:

- `train`: 60%
- `val`: 30%
- `test`: 10%

```bash
python -m src.pipelines.data_splitting
```

Optional flags:

- `--source-dir`
- `--target-dir`
- `--train-ratio`
- `--val-ratio`
- `--test-ratio`
- `--seed`
- `--overwrite`

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

The splitting script uses a fixed random seed by default (`42`) so the export
is reproducible.

## Entry Points

Available project entry points:

```bash
python main.py
python -m src.pipelines.data_preprocessing
python -m src.pipelines.data_splitting
python -m src.rag.ingest
python -m src.vision.train
python -m src.orchestrator.main
uvicorn app.main:app --reload
streamlit run app/streamlit.py
```

Currently, the dataset preparation scripts are implemented and runnable. Most
other application entry points are still placeholders.

## Development Notes

- Keep training and ML pipeline logic inside `src/`.
- Keep API and UI serving concerns inside `app/`.
- Load deployed model artifacts from `artifacts/`.
- Follow PEP 8 and prefer fully typed function signatures.
- Use concise module boundaries so training, serving, and orchestration remain
  separate.

## Suggested Next Steps

- Build the training and evaluation loop in `src/model/`.
- Add PDF or document ingestion in `src/rag/ingest.py`.
- Expose inference through `app/main.py` and `app/predict.py`.
- Connect prediction and retrieval in `src/orchestrator/main.py`.

## Project Metadata

- Package name: `rag-diagnostic-agent`
- Version: `0.1.0`
- Python requirement: `>=3.12`
