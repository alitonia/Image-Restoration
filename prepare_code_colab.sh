#!/bin/bash
# Script to package ONLY the source code for Google Colab

echo "📦 Preparing code for Google Colab..."

# Move to the root directory
cd "$(dirname "$0")"

# Zip ONLY the source code (Standard compression)
echo "📦 Packaging CODE (restoration_code.zip)..."
rm -f restoration_code.zip
zip -r restoration_code.zip training/ \
    -x "training/datasets/processed/*" \
    -x "training/datasets/raw/*" \
    -x "training/datasets/raw_lite/*" \
    -x "training/venv/*" \
    -x "**/__pycache__/*" \
    -x "**/.ipynb_checkpoints/*"

echo "✅ Code preparation complete! Please upload 'restoration_code.zip' to Google Drive."
