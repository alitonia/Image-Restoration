#!/bin/bash
# Script to process the full raw dataset

echo "Starting Full Data Preparation..."
./venv/bin/python scripts/prepare_data.py --input ./datasets/raw --output ./datasets/processed --seed 42
echo "Full Data Preparation Complete."
