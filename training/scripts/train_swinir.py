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
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import create_model

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class RestorationDataset(Dataset):
    def __init__(self, clean_dir, damaged_dir, patch_size=64, is_train=True):
        self.clean_files = sorted(glob(os.path.join(clean_dir, '*.*')))
        self.damaged_files = sorted(glob(os.path.join(damaged_dir, '*.*')))
        self.patch_size = patch_size
        self.is_train = is_train
        
        # Validation
        if len(self.clean_files) == 0:
            raise ValueError(f"No clean images found at {clean_dir}. Check your paths!")
        if len(self.damaged_files) == 0:
            raise ValueError(f"No damaged images found at {damaged_dir}. Check your paths!")
        
        print(f"📊 Dataset ({'Train' if is_train else 'Val'}) initialized: {len(self.clean_files)} pairs found.")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_img = cv2.imread(self.clean_files[idx])
        damaged_img = cv2.imread(self.damaged_files[idx])

        # Convert to RGB and Float
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        damaged_img = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.is_train:
            # Random Patch Extraction (for memory efficiency)
            h, w, _ = clean_img.shape
            x = np.random.randint(0, h - self.patch_size)
            y = np.random.randint(0, w - self.patch_size)
            clean_patch = clean_img[x:x+self.patch_size, y:y+self.patch_size, :]
            damaged_patch = damaged_img[x:x+self.patch_size, y:y+self.patch_size, :]
        else:
            # For validation, resize to multiple of 64
            h, w, _ = clean_img.shape
            target_h = (h // 64) * 64
            target_w = (w // 64) * 64
            clean_patch = cv2.resize(clean_img, (target_w, target_h))
            damaged_patch = cv2.resize(damaged_img, (target_w, target_h))

        # To Tensor (C, H, W)
        clean_patch = torch.from_numpy(np.transpose(clean_patch, (2, 0, 1)))
        damaged_patch = torch.from_numpy(np.transpose(damaged_patch, (2, 0, 1)))

        return damaged_patch, clean_patch

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--damaged_dir', type=str, required=True)
    parser.add_argument('--val_clean_dir', type=str, default=None)
    parser.add_argument('--val_damaged_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=8) # Small for 4GB VRAM
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--weight_path', type=str, default='../weights/swinir_restoration.pth')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    # set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device} | Seed: {args.seed}")
    
    if device.type == 'cpu':
        print("🛑 ERROR: NO GPU DETECTED. Colab -> Runtime -> Change Runtime Type -> T4 GPU.")

    # Model
    model = create_model(upscale=1).to(device) # upscale=1 for direct restoration
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Modern AMP syntax
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Datasets
    train_dataset = RestorationDataset(args.clean_dir, args.damaged_dir, args.patch_size, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_loader = None
    if args.val_clean_dir and args.val_damaged_dir:
        val_dataset = RestorationDataset(args.val_clean_dir, args.val_damaged_dir, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_psnr = -1.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for damaged, clean in pbar:
            damaged, clean = damaged.to(device), clean.to(device)

            optimizer.zero_grad()
            
            # Use AMP for memory efficiency
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(damaged)
                loss = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item() / args.batch_size})

        # Validation phase
        if val_loader:
            model.eval()
            val_psnr = 0
            with torch.no_grad():
                for damaged, clean in tqdm(val_loader, desc="Validating"):
                    damaged, clean = damaged.to(device), clean.to(device)
                    output = model(damaged)
                    val_psnr += calculate_psnr(output, clean).item()
            
            avg_psnr = val_psnr / len(val_loader)
            print(f"✅ Epoch {epoch+1} Val PSNR: {avg_psnr:.2f} dB")
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                os.makedirs(os.path.dirname(args.weight_path), exist_ok=True)
                torch.save(model.state_dict(), args.weight_path)
                print(f"🌟 New Best Model Saved (PSNR: {best_psnr:.2f})")
        else:
            # If no val set, just save the weights every epoch
            torch.save(model.state_dict(), args.weight_path)

        # Save checkpoint for progress tracking
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_psnr': best_psnr,
        }, checkpoint_path)

if __name__ == "__main__":
    train()
