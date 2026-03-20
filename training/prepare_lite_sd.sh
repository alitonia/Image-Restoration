#!/bin/bash
# Script to process a small set of images with Stable Diffusion optimized damage for quick validation

echo "Cleaning old processed SD-optimized data..."
rm -rf ./datasets/processed_sd

echo "Starting Lite SD Data Preparation..."
./venv/bin/python scripts/prepare_data_sd.py --input ./datasets/raw --output ./datasets/processed_sd --seed 42 --multiplier 10 --limit 10
echo "Lite SD Data Preparation Complete."
