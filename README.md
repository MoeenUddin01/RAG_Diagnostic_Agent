# RAG Diagnostic Agent

RAG Diagnostic Agent is a plant disease diagnosis project intended to combine
computer vision with retrieval-augmented generation (RAG). The target workflow
is straightforward: classify a plant image, then retrieve treatment guidance
from a curated knowledge base.

## Current State

This repository is a work in progress. The following components are implemented:

- **Data pipeline**: Dataset balancing, splitting, loading with `WeightedRandomSampler`
- **Model training**: Production-ready CLI pipeline with EfficientNet-B2 two-phase
  fine-tuning (frozen backbone warm-up + full fine-tuning), checkpointing,
  automatic model saving, progress bars with tqdm, and MLflow/DagsHub experiment tracking
- **Model evaluation**: Production-ready CLI pipeline with accuracy metrics,
  classification report, confusion matrix, and checkpoint loading
- **Experiment tracking**: MLflow integration with DagsHub for hyperparameter, metric, and artifact
  logging with visual experiment comparison and remote tracking
- **Utilities**: Device selection with CUDA compatibility checking (auto-fallback to CPU for incompatible GPUs),
  project root, artifacts directory helpers
- **Quick testing**: `max_batches` parameter in config.dev.yaml for fast smoke testing
- **API/UI**: Scaffolds only (not yet functional)
- **RAG**: Not yet implemented

The training and evaluation pipelines are fully functional and can be run via
command-line interfaces with full MLflow experiment tracking.

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
- MLflow
- DagsHub
- tqdm
- python-dotenv

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

### Environment Variables

For MLflow/DagsHub experiment tracking, create a `.env` file in the project root:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

The `.env` file is automatically loaded by `run.py` before MLflow initializes.

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
    use_weighted_sampler=True,  # Handles class imbalance
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

- **Train**: `WeightedRandomSampler` with inverse-frequency weights to handle class
  imbalance (e.g., `Potato___healthy` has only 91 samples vs 600 for most classes),
  `num_workers=4`, `pin_memory=True`. Set `use_weighted_sampler=False` for standard
  shuffling.
- **Val/Test**: `shuffle=False` (for consistent evaluation)

## Data Module Summary

| File            | Purpose                                | Key Function                                                               |
|-----------------|----------------------------------------|----------------------------------------------------------------------------|
| `transforms.py` | Image preprocessing pipelines          | `get_train_transform()`, `get_val_transform()`, `get_test_transform()`     |
| `dataset.py`    | Create PyTorch datasets from split folders | `get_datasets()` returns `ImageFolder` objects                             |
| `loader.py`     | Batch and iterate datasets for training  | `get_dataloaders()` returns `DataLoader` objects                         |
|                 |                                        | with optional `WeightedRandomSampler` for class imbalance                |

**Import chain:** `transforms.py` → `dataset.py` → `loader.py`

## Configuration

The project uses YAML configuration files for centralized settings management:

### config.yaml

Production configuration with default hyperparameters:

- **dataset**: Data paths, image size, class count, weighted sampler setting
- **model**: Architecture (EfficientNet-B2), pretrained weights, freeze epochs, classifier config
- **training**: Hyperparameters (30 epochs, batch size 32, learning rate, weight decay, patience)
- **artifacts**: Paths for checkpoints, history, and confusion matrix
- **rag**: Knowledge base and embedding model settings (for future RAG implementation)
- **mlflow**: Experiment tracking configuration
- **classes**: List of all 15 PlantVillage class names

### config.dev.yaml

Development configuration for fast smoke testing:

- **training**: 1 epoch, batch size 16, freeze_epochs=0, patience=1, max_batches=20 (limits total batches for quick testing)
- **artifacts**: Separate `artifacts/dev/` directory to avoid overwriting production artifacts
- **mlflow**: Uses `plant_disease_classifier_dev` experiment with `smoke-test` run name

Use `config.dev.yaml` to quickly verify the pipeline runs end-to-end without waiting for full training. The `max_batches` parameter limits the total number of batches processed across all epochs for rapid smoke testing.

### Running from Config

The project includes a `run.py` script that loads configuration from YAML files:

```bash
# Run with production config
python run.py config.yaml

# Run with dev/smoke-test config
python run.py config.dev.yaml
```

The `run.py` script automatically:

- Loads the specified YAML configuration
- Validates data directories exist
- Creates artifacts directory if needed
- Loads datasets with configured settings
- Trains the model with hyperparameters from config
- Logs all metrics to MLflow with experiment/run names from config
- Evaluates on validation set and prints results

## Model Training

Train the EfficientNet-B2 model on the prepared dataset using the production-ready
CLI pipeline:

```bash
python -m src.pipelines.model_training
```

### Training CLI Flags

| Flag                 | Type  | Default                   | Description                                          |
|----------------------|-------|---------------------------|------------------------------------------------------|
| `--train-dir`        | Path  | `dataset/processed/train` | Path to training data directory                      |
| `--val-dir`          | Path  | `dataset/processed/val`   | Path to validation data directory                    |
| `--batch-size`       | int   | `32`                      | Number of samples per batch                          |
| `--epochs`           | int   | `30`                      | Total number of training epochs                      |
| `--freeze-epochs`    | int   | `5`                       | Number of initial epochs with backbone frozen        |
| `--lr`               | float | `3e-4`                    | Initial learning rate for AdamW                      |
| `--weight-decay`     | float | `1e-4`                    | L2 regularization coefficient                        |
| `--patience`         | int   | `5`                       | Early stopping patience                              |
| `--seed`             | int   | `42`                      | Random seed for reproducibility                      |
| `--artifacts-dir`    | Path  | `artifacts`               | Directory for checkpoints and logs                   |
| `--no-weighted-sampler` | flag | `False`                | Disable weighted random sampling for class imbalance |
| `--experiment-name`  | str   | `plant_disease_classifier` | MLflow experiment name for tracking              |
| `--run-name`         | str   | `None`                    | Optional MLflow run name for this training run      |

### Example Usage

```bash
# Train with default settings
python -m src.pipelines.model_training

# Train with custom hyperparameters
python -m src.pipelines.model_training \
    --train-dir dataset/processed/train \
    --val-dir dataset/processed/val \
    --batch-size 64 \
    --epochs 50 \
    --freeze-epochs 10 \
    --lr 1e-4 \
    --weight-decay 5e-4 \
    --artifacts-dir artifacts/experiment1
```

### Training Process

The pipeline implements a two-phase fine-tuning strategy:

1. **Warm-up phase** (epochs 0 to `freeze-epochs` - 1): Backbone frozen, only the
   classification head trains
2. **Full fine-tuning phase** (epochs `freeze-epochs` to `epochs` - 1): All parameters
   unfrozen for end-to-end training

**Progress Tracking:**

Training includes real-time progress bars (using tqdm) that show batch-level progress for both training and validation phases, displaying percentage complete and estimated time remaining.

### Model Checkpoints

The training pipeline automatically saves two checkpoints to `artifacts/`:

- **best_model.pt**: Model with the lowest validation loss (saved when performance improves)
- **last_model.pt**: Model from the final epoch (saved regardless of performance)

Each checkpoint contains:

- `epoch`: Epoch index when saved
- `val_loss`: Validation loss at that epoch
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `class_names`: List of class label names (for evaluation)

### MLflow Experiment Tracking

The training pipeline automatically logs all experiments to MLflow with DagsHub integration for remote tracking:

**Logged Hyperparameters:**

- epochs, freeze_epochs, lr, weight_decay, patience, seed, batch_size, num_classes

**Logged Metrics (per epoch):**

- train_loss, train_acc, val_loss, val_acc

**Logged Summary Metrics:**

- best_val_loss, best_val_acc

**Logged Artifacts:**

- best_model.pt: Best model checkpoint
- history.json: Training history with per-epoch metrics

**DagsHub Integration:**

The training pipeline automatically initializes DagsHub tracking before MLflow experiment setup. Ensure your `.env` file contains the required DagsHub credentials (see Installation section). This enables remote experiment tracking and visualization on DagsHub.

**Viewing Experiments:**

After training, start the MLflow UI to visualize experiments locally:

```bash
mlflow ui --port 5000
```

Then open `http://localhost:5000` in your browser to compare runs, view metrics, and download artifacts. For remote viewing, check your DagsHub repository's Experiments tab.

## Model Evaluation

Evaluate a trained model on the test split using the production-ready CLI pipeline:

```bash
python -m src.pipelines.model_evaluation
```

### Evaluation CLI Flags

| Flag            | Type  | Default                    | Description                                    |
|-----------------|-------|----------------------------|------------------------------------------------|
| `--test-dir`    | Path  | `dataset/processed/test`   | Path to test data directory                    |
| `--checkpoint`  | Path  | `artifacts/best_model.pt`  | Path to model checkpoint file                  |
| `--batch-size`  | int   | `32`                       | Number of samples per batch                    |
| `--artifacts-dir` | Path | `artifacts`               | Directory for saving evaluation artifacts      |
| `--save-cm`     | flag  | `False`                    | Save confusion matrix as PNG                   |

### Evaluation Examples

```bash
# Evaluate with default settings
python -m src.pipelines.model_evaluation

# Evaluate a specific checkpoint and save confusion matrix
python -m src.pipelines.model_evaluation \
    --test-dir dataset/processed/test \
    --checkpoint artifacts/best_model.pt \
    --batch-size 64 \
    --save-cm \
    --artifacts-dir artifacts/evaluation
```

### Evaluation Output

The pipeline prints:

- Checkpoint information (epoch, validation loss)
- Evaluation configuration (device, classes, sample count)
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix (if `--save-cm` is enabled)

**Note**: The evaluation pipeline requires checkpoints saved with the updated
`model_training.py` that includes `class_names` in the checkpoint. Old checkpoints
will produce an error message instructing you to retrain.

## Entry Points

Available project entry points:

```bash
python main.py
python -m src.pipelines.data_preprocessing
python -m src.pipelines.data_splitting
python -m src.pipelines.model_training
python -m src.pipelines.model_evaluation
python -m src.rag.ingest
python -m src.vision.train
python -m src.orchestrator.main
uvicorn app.main:app --reload
streamlit run app/streamlit.py
```

The dataset preparation scripts (`data_preprocessing`, `data_splitting`) and
model training/evaluation pipelines (`model_training`, `model_evaluation`) are
implemented and runnable. API and UI entry points are still scaffolds.

## Development Notes

- Keep training and ML pipeline logic inside `src/`.
- Keep API and UI serving concerns inside `app/`.
- Load deployed model artifacts from `artifacts/`.
- Follow PEP 8 and prefer fully typed function signatures.
- Use concise module boundaries so training, serving, and orchestration remain
  separate.
- The `get_device()` utility automatically detects GPU compatibility and falls back to CPU if the GPU's compute capability is incompatible with the installed PyTorch version. This prevents runtime errors on older GPUs.

## Suggested Next Steps

- Implement inference script in `src/model/prediction.py`.
- Add PDF or document ingestion in `src/rag/ingest.py`.
- Expose inference through `app/main.py` and `app/predict.py`.
- Connect prediction and retrieval in `src/orchestrator/main.py`.

## Project Metadata

- Package name: `rag-diagnostic-agent`
- Version: `0.1.0`
- Python requirement: `>=3.12`
