#!/bin/bash
# Script to process a small set of images for quick validation

echo "Cleaning old processed data..."
rm -rf ./datasets/processed

echo "Starting Lite Data Preparation..."
./venv/bin/python scripts/prepare_data.py --input ./datasets/raw_lite --output ./datasets/processed --seed 42
echo "Lite Data Preparation Complete."
