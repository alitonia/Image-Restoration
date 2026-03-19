#!/bin/bash
# Script to package the training directory for Google Colab

echo "📦 Preparing project for Google Colab..."

# Move to the root directory
cd "$(dirname "$0")"

# 1. Zip ONLY the processed data (No compression - FAST)
echo "📦 Packaging DATASET (processed_data.zip)..."
rm -f processed_data.zip
if [ -d "training/datasets/processed" ]; then
    zip -r processed_data.zip training/datasets/processed/
else
    echo "⚠️ Warning: training/datasets/processed not found. Skipping data zip."
fi

# 2. Zip ONLY the source code (Standard compression)
echo "📦 Packaging CODE (restoration_code.zip)..."
rm -f restoration_code.zip
zip -r restoration_code.zip training/ \
    -x "training/datasets/processed/*" \
    -x "training/datasets/raw/*" \
    -x "training/datasets/raw_lite/*" \
    -x "training/venv/*" \
    -x "**/__pycache__/*" \
    -x "**/.ipynb_checkpoints/*"

echo "✅ Preparation complete!"
echo "🚀 Next steps:"
echo "1. Upload BOTH 'processed_data.zip' and 'restoration_code.zip' to Google Drive."
echo "2. Open 'training/colab_training.ipynb' in Colab and follow the instructions."
