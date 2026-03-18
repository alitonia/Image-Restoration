#!/bin/bash
# Script to process the full raw dataset

echo "Cleaning old processed data..."
rm -rf ./datasets/processed

echo "Starting Full Data Preparation..."
./venv/bin/python scripts/prepare_data.py --input ./datasets/raw --output ./datasets/processed --seed 42 --multiplier 1
echo "Full Data Preparation Complete."
