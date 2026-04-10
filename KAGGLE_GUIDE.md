# Kaggle Training Guide for RAG Diagnostic Agent

This guide shows you how to train your plant disease classification model on Kaggle using GPU acceleration.

## Why Train on Kaggle?

- **Free GPU access**: T4 or V100 GPUs for faster training
- **No local resources**: Doesn't use your local machine's compute
- **Easy dataset access**: PlantVillage dataset is readily available
- **Collaboration**: Easy to share and reproduce experiments

## Prerequisites

1. Kaggle account (free): https://www.kaggle.com/
2. Your project source code from this repository

## Method 1: Using Kaggle Notebooks (Recommended)

### Step 1: Prepare Your Code

Create a compressed archive of your source code:

```bash
# From your project root
cd /home/moeenuddin/Desktop/Deep_learning/RAGAge/RAG_Diagnostic_Agent
tar -czf src_code.tar.gz src/ run.py config.yaml
```

### Step 2: Create a Kaggle Dataset

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload `src_code.tar.gz`
4. Title: "RAG Diagnostic Agent Source Code"
5. Set visibility to "Public" or "Private"
6. Click "Create"

### Step 3: Create a Kaggle Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. **Important**: Set "Accelerator" to "GPU" (T4 or V100)
4. Choose "Python" as the language

### Step 4: Add Datasets to Notebook

1. In the right sidebar, click "Add data"
2. Search for and add:
   - **PlantVillage dataset**: Search "plantvillage" or "emmarex/plant-disease"
   - **Your source code**: Find the dataset you created in Step 2

### Step 5: Set Up the Notebook

Add this as the first cell in your Kaggle notebook:

```python
# Cell 1: Setup and install dependencies
import subprocess
import sys
from pathlib import Path

# Install required packages
packages = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "mlflow>=2.13.0",
    "python-dotenv>=1.0.0",
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✓ Dependencies installed")
```

```python
# Cell 2: Set up project structure
import shutil
from pathlib import Path

# Define paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
PROJECT_ROOT = KAGGLE_WORKING / "rag-diagnostic-agent"

# Create directories
PROJECT_ROOT.mkdir(exist_ok=True)
(PROJECT_ROOT / "dataset" / "raw").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "dataset" / "processed").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

print(f"✓ Project structure created at {PROJECT_ROOT}")
```

```python
# Cell 3: Extract source code
# Find your source code dataset (adjust the path as needed)
source_datasets = list(KAGGLE_INPUT.glob("*"))
source_dataset = None

for d in source_datasets:
    if "rag" in d.name.lower() or "diagnostic" in d.name.lower():
        source_dataset = d
        break

if source_dataset:
    # Extract the archive
    import tarfile
    archive_path = list(source_dataset.glob("*.tar.gz"))[0]
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(PROJECT_ROOT)
    print(f"✓ Source code extracted from {archive_path}")
else:
    print("⚠ Source code dataset not found. Please add it to the notebook.")
```

```python
# Cell 4: Copy PlantVillage dataset
# Find PlantVillage dataset (adjust the path as needed)
plant_datasets = list(KAGGLE_INPUT.glob("*"))
plant_dataset = None

for d in plant_datasets:
    if "plant" in d.name.lower():
        plant_dataset = d
        break

if plant_dataset:
    raw_dir = PROJECT_ROOT / "dataset" / "raw" / "PlantVillage"
    if not raw_dir.exists():
        shutil.copytree(plant_dataset, raw_dir, dirs_exist_ok=True)
    print(f"✓ PlantVillage dataset copied to {raw_dir}")
else:
    print("⚠ PlantVillage dataset not found. Please add it to the notebook.")
```

```python
# Cell 5: Update config for Kaggle paths
import yaml

config = {
    "dataset": {
        "train_dir": str(PROJECT_ROOT / "dataset" / "processed" / "train"),
        "val_dir": str(PROJECT_ROOT / "dataset" / "processed" / "val"),
        "test_dir": str(PROJECT_ROOT / "dataset" / "processed" / "test"),
        "num_classes": 15,
        "image_size": 224,
        "use_weighted_sampler": True,
    },
    "model": {
        "architecture": "efficientnet_b2",
        "pretrained": True,
        "freeze_epochs": 5,
        "classifier_dropout": 0.3,
        "classifier_in_features": 1408,
    },
    "training": {
        "epochs": 30,
        "batch_size": 32,  # Reduce to 16 if you get OOM errors
        "freeze_epochs": 5,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "patience": 5,
        "seed": 42,
    },
    "artifacts": {
        "dir": str(PROJECT_ROOT / "artifacts"),
        "best_model": str(PROJECT_ROOT / "artifacts" / "best_model.pt"),
        "last_model": str(PROJECT_ROOT / "artifacts" / "last_model.pt"),
        "history": str(PROJECT_ROOT / "artifacts" / "history.json"),
        "confusion_matrix": str(PROJECT_ROOT / "artifacts" / "confusion_matrix.png"),
    },
    "mlflow": {
        "experiment_name": "plant_disease_classifier_kaggle",
        "tracking_uri": str(PROJECT_ROOT / "mlruns"),
        "run_name": "kaggle_gpu_run",
    },
    "classes": [
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
    ],
}

config_path = PROJECT_ROOT / "config.kaggle.yaml"
with config_path.open("w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✓ Configuration saved to {config_path}")
```

```python
# Cell 6: Run data preprocessing
import sys
import os

# Change to project directory
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Run preprocessing
print("Running data balancing...")
!python -m src.pipelines.data_preprocessing \
    --source-dir dataset/raw/PlantVillage \
    --target-dir dataset/raw/balance_dataset \
    --max-images-per-class 1000 \
    --overwrite

print("\nRunning data splitting...")
!python -m src.pipelines.data_splitting \
    --source-dir dataset/raw/balance_dataset \
    --target-dir dataset/processed \
    --train-ratio 0.6 \
    --val-ratio 0.3 \
    --test-ratio 0.1 \
    --overwrite
```

```python
# Cell 7: Run training
print("Starting model training...")
!python run.py config.kaggle.yaml
```

### Step 6: Download Results

After training completes:

1. Go to the "Output" tab in the right sidebar
2. Download:
   - `artifacts/best_model.pt` - Your trained model
   - `artifacts/history.json` - Training metrics
   - `artifacts/confusion_matrix.png` - Evaluation visualization

## Method 2: Using Kaggle Kernels (Script Mode)

For more advanced users, you can use Kaggle's script mode:

1. Create a script file with all the training code
2. Submit it as a Kaggle Kernel
3. It will run in the cloud and save outputs

See `kaggle_train.py` in this repository for a template.

## Common Issues and Solutions

### Out of Memory (OOM) Error

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in config:

```yaml
training:
  batch_size: 16  # Try 16 or 8 instead of 32
```

### Dataset Not Found

**Symptom**: `Error: Training directory not found`

**Solution**: Ensure you've:
1. Added the PlantVillage dataset to your notebook
2. Copied it to the correct location
3. Run the data preprocessing steps

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure you've:
1. Extracted your source code archive
2. Changed to the project directory with `os.chdir()`
3. Added the project root to `sys.path`

### Slow Training

**Solution**: 
- Ensure GPU accelerator is enabled
- Reduce `max_batches` in config for testing
- Use the development config for faster iterations

## Optimizing for Kaggle

### GPU Settings

Kaggle offers different GPU options:
- **T4** (default): Good for most models
- **V100**: Faster, but limited weekly hours
- **P100**: Older but still capable

### Batch Size Optimization

Start with batch size 32. If you get OOM errors:
- Try 16
- Try 8 (slowest but most stable)

### Epoch Configuration

For quick testing on Kaggle:
```yaml
training:
  epochs: 10  # Reduce for testing
  freeze_epochs: 2
```

For final training:
```yaml
training:
  epochs: 30  # Full training
  freeze_epochs: 5
```

## Alternative: Using Kaggle Datasets Directly

Instead of uploading your own data, you can use publicly available PlantVillage datasets:

1. **emmarex/plant-disease**: Most common
2. **rashmii/plantvillage-dataset**: Alternative source
3. **abdallahalide/plant-village**: Another option

Add the dataset in the notebook's "Add data" section and adjust the copy path accordingly.

## Saving and Sharing Your Work

### Save the Notebook

1. Click "Save Version" in the top right
2. Choose "Save & Run All" (for a complete run)
3. Or "Quick Save" (to save without running)

### Share the Notebook

1. Go to your notebook's page
2. Click "Share"
3. Copy the link and share it

### Download the Notebook

1. Click "Save Version"
2. After saving, click the three dots next to the version
3. Select "Download"

## Next Steps After Kaggle Training

1. **Download your model**: Get `best_model.pt` from the Output tab
2. **Move to local project**: Copy it to your local `artifacts/` directory
3. **Run evaluation**: Use the evaluation pipeline on your local machine
4. **Deploy**: Use the trained model in your FastAPI or Streamlit app

## Additional Resources

- [Kaggle Documentation](https://www.kaggle.com/docs)
- [PyTorch on Kaggle](https://www.kaggle.com/docs/technical-notebooks#gpu)
- [PlantVillage Dataset Paper](https://arxiv.org/abs/1511.08060)

## Troubleshooting Checklist

- [ ] GPU accelerator is enabled
- [ ] All dependencies are installed
- [ ] Source code is extracted and in `sys.path`
- [ ] PlantVillage dataset is copied to correct location
- [ ] Data preprocessing completed successfully
- [ ] Config paths are correct for Kaggle environment
- [ ] Batch size is appropriate for GPU memory
- [ ] Working directory is set to project root

## Support

If you encounter issues:
1. Check the Kaggle notebook's output logs
2. Verify all paths are correct
3. Ensure GPU is enabled in notebook settings
4. Check that dataset is properly added to notebook
