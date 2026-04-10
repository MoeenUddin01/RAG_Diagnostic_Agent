#!/usr/bin/env python3
"""Kaggle training script for RAG Diagnostic Agent.

This script is designed to run on Kaggle Notebooks with GPU acceleration.
It handles dataset download, preprocessing, and model training in one run.

Usage on Kaggle:
    1. Create a new Notebook on Kaggle
    2. Set accelerator to GPU (T4 or V100)
    3. Upload this script as a code cell or as a file
    4. Run the script
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Kaggle-specific paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_DATASET_NAME = "plantvillage"  # Common Kaggle dataset name


def install_dependencies() -> None:
    """Install required dependencies for Kaggle environment."""
    print("=" * 60)
    print("Installing dependencies...")
    print("=" * 60)

    # Core ML dependencies
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
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("✓ Dependencies installed successfully\n")


def setup_project_structure() -> Path:
    """Set up project directory structure in Kaggle working directory.

    Returns:
        Path to the project root in Kaggle working directory.
    """
    print("=" * 60)
    print("Setting up project structure...")
    print("=" * 60)

    # Create project directories
    project_root = KAGGLE_WORKING / "rag-diagnostic-agent"
    project_root.mkdir(exist_ok=True)

    (project_root / "dataset" / "raw").mkdir(parents=True, exist_ok=True)
    (project_root / "dataset" / "processed").mkdir(parents=True, exist_ok=True)
    (project_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_root / "src").mkdir(exist_ok=True)

    print(f"✓ Project structure created at {project_root}\n")
    return project_root


def download_plantvillage_dataset(project_root: Path) -> Path:
    """Download PlantVillage dataset from Kaggle.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Path to the downloaded dataset directory.
    """
    print("=" * 60)
    print("Downloading PlantVillage dataset...")
    print("=" * 60)

    # Check if dataset already exists in Kaggle input
    # PlantVillage is available as: emmarex/plant-disease or similar
    # We'll use the most common one

    # List available PlantVillage datasets
    input_dirs = list(KAGGLE_INPUT.iterdir()) if KAGGLE_INPUT.exists() else []

    plantvillage_path = None

    # Try to find PlantVillage dataset in input
    for input_dir in input_dirs:
        if "plant" in input_dir.name.lower():
            print(f"Found potential dataset: {input_dir}")
            plantvillage_path = input_dir
            break

    if plantvillage_path is None:
        print("PlantVillage dataset not found in /kaggle/input")
        print("Please add a dataset to your Kaggle notebook:")
        print("  Search for 'PlantVillage' or 'plant disease' on Kaggle Datasets")
        print("  Common dataset: 'emmarex/plant-disease' or 'rashmii/plantvillage-dataset'")
        print("\nAfter adding the dataset, re-run this script.")
        sys.exit(1)

    # Copy dataset to project raw directory
    raw_dir = project_root / "dataset" / "raw"
    target_dir = raw_dir / "PlantVillage"

    if target_dir.exists():
        print(f"✓ Dataset already exists at {target_dir}")
    else:
        print(f"Copying dataset from {plantvillage_path} to {target_dir}")
        if plantvillage_path.is_dir():
            shutil.copytree(plantvillage_path, target_dir, dirs_exist_ok=True)
        else:
            print(f"Error: {plantvillage_path} is not a directory")
            sys.exit(1)

    print(f"✓ Dataset ready at {target_dir}\n")
    return target_dir


def create_kaggle_config(project_root: Path) -> Path:
    """Create Kaggle-optimized configuration file.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Path to the created config file.
    """
    print("=" * 60)
    print("Creating Kaggle-optimized configuration...")
    print("=" * 60)

    config = {
        "dataset": {
            "train_dir": str(project_root / "dataset" / "processed" / "train"),
            "val_dir": str(project_root / "dataset" / "processed" / "val"),
            "test_dir": str(project_root / "dataset" / "processed" / "test"),
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
            "batch_size": 32,  # Adjust based on GPU memory
            "freeze_epochs": 5,
            "lr": 0.0003,
            "weight_decay": 0.0001,
            "patience": 5,
            "seed": 42,
        },
        "artifacts": {
            "dir": str(project_root / "artifacts"),
            "best_model": str(project_root / "artifacts" / "best_model.pt"),
            "last_model": str(project_root / "artifacts" / "last_model.pt"),
            "history": str(project_root / "artifacts" / "history.json"),
            "confusion_matrix": str(project_root / "artifacts" / "confusion_matrix.png"),
        },
        "mlflow": {
            "experiment_name": "plant_disease_classifier_kaggle",
            "tracking_uri": str(project_root / "mlruns"),
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

    config_path = project_root / "config.kaggle.yaml"
    import yaml

    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✓ Configuration saved to {config_path}\n")
    return config_path


def create_source_files(project_root: Path) -> None:
    """Create necessary source files in the project.

    Args:
        project_root: Path to the project root directory.
    """
    print("=" * 60)
    print("Creating source files...")
    print("=" * 60)

    # Create minimal source files needed for training
    # In a real scenario, you would upload your actual src/ directory

    src_dir = project_root / "src"

    # Create __init__.py files
    (src_dir / "__init__.py").write_text("")
    (src_dir / "data" / "__init__.py").write_text("")
    (src_dir / "model" / "__init__.py").write_text("")
    (src_dir / "pipelines" / "__init__.py").write_text("")

    print("⚠ Note: You need to upload your actual src/ directory files")
    print("  Recommended: Create a Kaggle Dataset with your src/ directory")
    print("  Then add it as input to your notebook\n")


def run_data_preprocessing(project_root: Path) -> None:
    """Run data preprocessing pipeline.

    Args:
        project_root: Path to the project root directory.
    """
    print("=" * 60)
    print("Running data preprocessing...")
    print("=" * 60)

    # This would run your data_preprocessing.py
    # For now, we'll provide instructions
    print("To run data preprocessing, ensure you have:")
    print("  1. src/data/balancing.py")
    print("  2. src/data/splitting.py")
    print("  3. src/pipelines/data_preprocessing.py")
    print("  4. src/pipelines/data_splitting.py")
    print("\nThen run:")
    print(f"  cd {project_root}")
    print("  python -m src.pipelines.data_preprocessing")
    print("  python -m src.pipelines.data_splitting\n")


def run_training(project_root: Path, config_path: Path) -> None:
    """Run model training.

    Args:
        project_root: Path to the project root directory.
        config_path: Path to the configuration file.
    """
    print("=" * 60)
    print("Starting model training...")
    print("=" * 60)

    # This would run your training pipeline
    print("To run training, ensure you have:")
    print("  1. All source files from src/ uploaded")
    print("  2. run.py uploaded")
    print("\nThen run:")
    print(f"  cd {project_root}")
    print(f"  python run.py {config_path.name}\n")


def main() -> None:
    """Main execution function."""
    print("\n" + "=" * 60)
    print("  RAG Diagnostic Agent - Kaggle Training Setup")
    print("=" * 60 + "\n")

    # Step 1: Install dependencies
    install_dependencies()

    # Step 2: Setup project structure
    project_root = setup_project_structure()

    # Step 3: Download dataset
    dataset_path = download_plantvillage_dataset(project_root)

    # Step 4: Create configuration
    config_path = create_kaggle_config(project_root)

    # Step 5: Create source files placeholder
    create_source_files(project_root)

    # Step 6: Instructions for next steps
    print("=" * 60)
    print("  Setup Complete! Next Steps:")
    print("=" * 60)
    print("\n1. Upload your source code:")
    print("   - Create a Kaggle Dataset with your src/ directory")
    print("   - Upload run.py to the notebook")
    print("   - Add the dataset as input to your notebook")
    print("\n2. Update dataset path in config if needed")
    print("\n3. Run preprocessing:")
    print(f"   cd {project_root}")
    print("   python -m src.pipelines.data_preprocessing")
    print("   python -m src.pipelines.data_splitting")
    print("\n4. Run training:")
    print(f"   python run.py {config_path.name}")
    print("\n5. Download artifacts:")
    print(f"   Models will be saved to {project_root / 'artifacts'}")
    print("   Download them from the Output tab")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
