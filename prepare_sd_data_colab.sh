#!/bin/bash
# Script to package the SD-optimized processed data for Google Colab

echo "📦 Packaging SD-Optimized DATASET (processed_sd.zip)..."

# Move to the root directory
cd "$(dirname "$0")"

rm -f processed_sd.zip

if [ -d "training/datasets/processed_sd" ]; then
    # Zipping without compression is much faster for large datasets in Colab
    zip -0 -r processed_sd.zip training/datasets/processed_sd/
    echo "✅ Packaging complete: processed_sd.zip created."
    echo "🚀 Upload 'processed_sd.zip' to the root of your Google Drive."
else
    echo "❌ Error: training/datasets/processed_sd not found. Run prepare_lite_sd.sh first."
    exit 1
fi
