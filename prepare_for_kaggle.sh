#!/bin/bash
# Prepare source code for upload to Kaggle

set -e

echo "========================================"
echo "Preparing code for Kaggle upload"
echo "========================================"

# Create a temporary directory for the Kaggle package
KAGGLE_DIR="kaggle_package"
rm -rf "$KAGGLE_DIR"
mkdir -p "$KAGGLE_DIR"

# Copy essential files
echo "Copying source code..."
cp -r src/ "$KAGGLE_DIR/"
cp run.py "$KAGGLE_DIR/"
cp config.yaml "$KAGGLE_DIR/"
cp config.dev.yaml "$KAGGLE_DIR/"

# Create a README for the Kaggle dataset
cat > "$KAGGLE_DIR/README.md" << 'EOF'
# RAG Diagnostic Agent - Source Code

This package contains the source code for training the plant disease classification model on Kaggle.

## Contents

- `src/`: Source code modules
- `run.py`: Main training script
- `config.yaml`: Production configuration
- `config.dev.yaml`: Development configuration

## Usage on Kaggle

1. Extract this archive to `/kaggle/working/rag-diagnostic-agent/`
2. Follow the KAGGLE_GUIDE.md instructions for setup
3. Run training with: `python run.py config.yaml`
EOF

# Create the archive
echo "Creating archive..."
tar -czf kaggle_source_code.tar.gz -C "$KAGGLE_DIR" .

# Clean up
rm -rf "$KAGGLE_DIR"

echo "========================================"
echo "✓ Archive created: kaggle_source_code.tar.gz"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Upload kaggle_source_code.tar.gz to Kaggle as a Dataset"
echo "2. Add the dataset to your Kaggle Notebook"
echo "3. Follow KAGGLE_GUIDE.md to complete setup"
echo ""
