#!/bin/bash
# Script to package the training directory for Google Colab

echo "📦 Preparing project for Google Colab..."

# Move to the root directory
cd "$(dirname "$0")"

# Remove old zip if it exists
rm -f restoration_training.zip

# Create the zip file
# Excludes: venv (too large), processed data (can be regenerated or uploaded separately), 
# and git files.
zip -r restoration_training.zip training/ \
    -x "training/venv/*" \
    -x "training/datasets/processed/*" \
    -x "training/datasets/raw/*" \
    -x "**/__pycache__/*" \
    -x "**/.ipynb_checkpoints/*"

echo "✅ Preparation complete!"
echo "🚀 Next steps:"
echo "1. Download 'restoration_training.zip' to your local machine."
echo "2. Upload it to the root of your Google Drive."
echo "3. Open 'training/colab_training.ipynb' in Colab and follow the instructions."
