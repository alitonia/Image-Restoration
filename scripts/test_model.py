import torch
import sys
import os
import cv2
sys.path.append('/home/alitonia/work/Image-Restoration/training')
from src.model import create_model
from scripts.train_swinir import RestorationDataset

def test_on_real_dataset():
    clean_dir = '/home/alitonia/work/Image-Restoration/training/datasets/processed/train/clean'
    damaged_dir = '/home/alitonia/work/Image-Restoration/training/datasets/processed/train/damaged'
    
    # Check if dataset exists
    if not os.path.exists(clean_dir):
        print(f"Skipping real dataset test, {clean_dir} does not exist")
        return
        
    # 1. Test training mode (patch extraction)
    print("\n--- Testing Training Dataset (is_train=True, patch_size=64) ---")
    dataset_train = RestorationDataset(clean_dir, damaged_dir, patch_size=64, is_train=True)
    damaged, clean = dataset_train[0]
    print(f"Damaged shape: {damaged.shape}")
    print(f"Clean shape: {clean.shape}")
    
    model = create_model()
    # add batch dimension
    damaged_batch = damaged.unsqueeze(0)
    try:
        out = model(damaged_batch)
        print(f"Model output shape (train): {out.shape}")
        print("✅ Training pass succeeded on real data!")
    except Exception as e:
        print(f"❌ Error during training forward pass: {e}")

    # 2. Test validation mode (full resized image)
    print("\n--- Testing Validation Dataset (is_train=False) ---")
    dataset_val = RestorationDataset(clean_dir, damaged_dir, is_train=False)
    damaged_val, clean_val = dataset_val[0]
    print(f"Damaged shape before model: {damaged_val.shape}")
    
    damaged_val_batch = damaged_val.unsqueeze(0)
    try:
        out_val = model(damaged_val_batch)
        print(f"Model output shape (val): {out_val.shape}")
        print("✅ Validation pass succeeded on real data!")
    except Exception as e:
        print(f"❌ Error during validation forward pass: {e}")

if __name__ == '__main__':
    test_on_real_dataset()
