import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import create_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure reproducible results even on multi-worker loaders (optional but good)
    os.environ['PYTHONHASHSEED'] = str(seed)

class RestorationDataset(Dataset):
    def __init__(self, clean_dir, damaged_dir, patch_size=64):
        self.clean_files = sorted(glob(os.path.join(clean_dir, '*.*')))
        self.damaged_files = sorted(glob(os.path.join(damaged_dir, '*.*')))
        self.patch_size = patch_size
        assert len(self.clean_files) == len(self.damaged_files), "Mismatch between clean and damaged files"

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_img = cv2.imread(self.clean_files[idx])
        damaged_img = cv2.imread(self.damaged_files[idx])

        # Convert to RGB and Float
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        damaged_img = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Random Patch Extraction (for memory efficiency)
        h, w, _ = clean_img.shape
        x = np.random.randint(0, h - self.patch_size)
        y = np.random.randint(0, w - self.patch_size)

        clean_patch = clean_img[x:x+self.patch_size, y:y+self.patch_size, :]
        damaged_patch = damaged_img[x:x+self.patch_size, y:y+self.patch_size, :]

        # To Tensor (C, H, W)
        clean_patch = torch.from_numpy(np.transpose(clean_patch, (2, 0, 1)))
        damaged_patch = torch.from_numpy(np.transpose(damaged_patch, (2, 0, 1)))

        return damaged_patch, clean_patch

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--damaged_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=8) # Small for 4GB VRAM
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--weight_path', type=str, default='../weights/swinir_restoration.pth')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device} | Seed: {args.seed}")

    # Model
    model = create_model(upscale=1).to(device) # upscale=1 for direct restoration
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() # For Mixed Precision (important for RTX 2050)

    start_epoch = 0
    checkpoint_path = args.weight_path.replace('.pth', '_checkpoint.pth')
    
    # Resume logic
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch: {start_epoch}")
    elif os.path.exists(args.weight_path) and args.resume:
        print(f"Loading weights only: {args.weight_path}")
        model.load_state_dict(torch.load(args.weight_path, map_location=device))

    dataset = RestorationDataset(args.clean_dir, args.damaged_dir, args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for damaged, clean in pbar:
            damaged, clean = damaged.to(device), clean.to(device)

            optimizer.zero_grad()
            
            # Use AMP for memory efficiency
            with torch.cuda.amp.autocast():
                output = model(damaged)
                loss = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        # Save Best and Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(args.weight_path), exist_ok=True)
            torch.save(model.state_dict(), args.weight_path)
            
            # Save full checkpoint for resuming
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Model and checkpoint saved at epoch {epoch+1}.")

if __name__ == "__main__":
    train()
