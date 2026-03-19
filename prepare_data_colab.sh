#!/bin/bash
# Script to package ONLY the training dataset for Google Colab

echo "📦 Preparing data for Google Colab..."

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

echo "✅ Data preparation complete! Please upload 'processed_data.zip' to Google Drive."
