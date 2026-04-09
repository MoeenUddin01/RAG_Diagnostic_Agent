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
- `src/data/transforms.py` for image preprocessing and augmentation
- `src/data/dataset.py` for PyTorch ImageFolder dataset creation
- `src/data/loader.py` for DataLoader creation with batching and shuffling

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

**Class-wise distribution:**

| Class | Train | Val | Test |
|-------|-------|-----|------|
| Pepper__bell___Bacterial_spot | 598 | 299 | 100 |
| Pepper__bell___healthy | 600 | 300 | 100 |
| Potato___Early_blight | 600 | 300 | 100 |
| Potato___Late_blight | 600 | 300 | 100 |
| Potato___healthy | 91 | 45 | 16 |
| Tomato_Bacterial_spot | 600 | 300 | 100 |
| Tomato_Early_blight | 600 | 300 | 100 |
| Tomato_Late_blight | 600 | 300 | 100 |
| Tomato_Leaf_Mold | 571 | 285 | 96 |
| Tomato_Septoria_leaf_spot | 600 | 300 | 100 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 600 | 300 | 100 |
| Tomato__Target_Spot | 600 | 300 | 100 |
| Tomato__Tomato_YellowLeaf__Curl_Virus | 600 | 300 | 100 |
| Tomato__Tomato_mosaic_virus | 223 | 111 | 39 |
| Tomato_healthy | 600 | 300 | 100 |
| **TOTAL** | **8083** | **4040** | **1351** |

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

### 3. Load data for training

After splitting, use the data loading utilities to create PyTorch datasets and
DataLoaders:

```python
from src.data import get_dataloaders, get_datasets

# Option 1: Get DataLoaders directly (recommended for training)
train_loader, val_loader, test_loader = get_dataloaders(
    train_dir="dataset/processed/train",
    val_dir="dataset/processed/val",
    test_dir="dataset/processed/test",
    batch_size=32,
)

# Option 2: Get datasets only (for custom DataLoader setup)
train_ds, val_ds, test_ds = get_datasets(
    train_dir="dataset/processed/train",
    val_dir="dataset/processed/val",
    test_dir="dataset/processed/test",
)
```

The data loading pipeline applies these transforms:

| Split | Transforms                                                                 |
|-------|----------------------------------------------------------------------------|
| Train | Resize(224,224) → RandomHorizontalFlip → ToTensor → Normalize(ImageNet)    |
| Val   | Resize(224,224) → ToTensor → Normalize(ImageNet)                           |
| Test  | Resize(224,224) → ToTensor → Normalize(ImageNet)                           |

DataLoaders configuration:

- **Train**: `shuffle=True` (for parameter updates), `num_workers=4`, `pin_memory=True`
- **Val/Test**: `shuffle=False` (for consistent evaluation)

## Data Module Summary

| File            | Purpose                                | Key Function                                                               |
|-----------------|----------------------------------------|----------------------------------------------------------------------------|
| `transforms.py` | Image preprocessing pipelines          | `get_train_transform()`, `get_val_transform()`, `get_test_transform()`     |
| `dataset.py`    | Create PyTorch datasets from split folders | `get_datasets()` returns `ImageFolder` objects                             |
| `loader.py`     | Batch and iterate datasets for training | `get_dataloaders()` returns `DataLoader` objects                           |

**Import chain:** `transforms.py` → `dataset.py` → `loader.py`

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
