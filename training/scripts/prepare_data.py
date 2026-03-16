import cv2
import numpy as np
import os
import random
import hashlib
from tqdm import tqdm
from pathlib import Path

def get_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_random_mask(h, w):
    """Generates a random mask (0 to 1) for localized damage."""
    mask_type = random.choice(['full', 'rect', 'ellipse', 'multiply'])
    mask = np.ones((h, w, 1), dtype=np.float32)
    
    if mask_type == 'full':
        return mask
    
    if mask_type == 'rect':
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(x1 + 10, w), random.randint(y1 + 10, h)
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        sub_mask[y1:y2, x1:x2, :] = 1.0
        return sub_mask
        
    if mask_type == 'ellipse':
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(20, w), random.randint(20, h))
        angle = random.randint(0, 360)
        cv2.ellipse(sub_mask, center, axes, angle, 0, 360, 1.0, -1)
        return sub_mask

    if mask_type == 'multiply':
        # Multiple small patches
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(2, 5)):
              center = (random.randint(0, w), random.randint(0, h))
              axes = (random.randint(10, w//4), random.randint(10, h//4))
              cv2.ellipse(sub_mask, center, axes, random.randint(0, 360), 0, 360, 1.0, -1)
        return sub_mask
    
    return mask

def add_noise(img, localized=True):
    """Add Gaussian noise to the image, potentially localized."""
    h, w, c = img.shape
    var = random.uniform(0.001, 0.05)
    gauss = np.random.normal(0, var**0.5, (h, w, c)).astype(np.float32)
    
    if localized:
        mask = get_random_mask(h, w)
        img_noise = img + gauss * mask
    else:
        img_noise = img + gauss
        
    return np.clip(img_noise, 0, 1)

def add_blur(img, localized=True):
    """Add Gaussian blur, potentially localized."""
    h, w, _ = img.shape
    kernel_size = random.choice([3, 5, 7, 9])
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    if localized:
        mask = get_random_mask(h, w)
        img_blur = blurred * mask + img * (1 - mask)
    else:
        img_blur = blurred
        
    return np.clip(img_blur, 0, 1)

def add_scratches(img):
    """Simulate random scratches (already localized by nature)."""
    damaged_img = img.copy()
    num_scratches = random.randint(3, 15)
    for _ in range(num_scratches):
        x1, y1 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
        x2, y2 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
        thickness = random.randint(1, 2)
        color = (random.uniform(0.5, 0.9), random.uniform(0.5, 0.9), random.uniform(0.5, 0.9))
        cv2.line(damaged_img, (x1, y1), (x2, y2), color, thickness)
    return damaged_img

def add_yellowing(img, localized=True):
    """Aging/Yellowing shift."""
    img_yellow = img.copy()
    # BGR format: 0=Blue, 1=Green, 2=Red
    img_yellow[:, :, 0] *= random.uniform(0.5, 0.7)  # Reduce Blue
    img_yellow[:, :, 2] *= random.uniform(1.0, 1.1)  # Slight Red boost
    
    if localized:
        mask = get_random_mask(img.shape[0], img.shape[1])
        img_res = img_yellow * mask + img * (1 - mask)
    else:
        img_res = img_yellow
        
    return np.clip(img_res, 0, 1)

def save_pair(clean_img_uint8, damaged_img_float, output_path, name_prefix, suffix, file_ext):
    clean_dir = output_path / 'clean'
    damaged_dir = output_path / 'damaged'
    target_name = f"{name_prefix}_{suffix}{file_ext}"
    cv2.imwrite(str(clean_dir / target_name), clean_img_uint8)
    damaged_img_uint8 = (damaged_img_float * 255).astype(np.uint8)
    cv2.imwrite(str(damaged_dir / target_name), damaged_img_uint8)

def process_images_recursive(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    (output_path / 'clean').mkdir(parents=True, exist_ok=True)
    (output_path / 'damaged').mkdir(parents=True, exist_ok=True)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.rglob(ext))
    
    print(f"Found {len(image_paths)} images. Generating localized damages...")
    
    processed_hashes = set()
    for path in tqdm(image_paths):
        file_hash = get_file_hash(path)
        if file_hash in processed_hashes: continue
        processed_hashes.add(file_hash)

        img = cv2.imread(str(path))
        if img is None: continue
        
        img_float = img.astype(np.float32) / 255.0
        
        # 1. Localized Blur
        save_pair(img, add_blur(img_float.copy()), output_path, file_hash, "blur", path.suffix)
        
        # 2. Localized Noise
        save_pair(img, add_noise(img_float.copy()), output_path, file_hash, "noise", path.suffix)
        
        # 3. Mixed Realistic
        damaged_mixed = add_blur(img_float.copy(), localized=random.choice([True, False]))
        damaged_mixed = add_noise(damaged_mixed, localized=random.choice([True, False]))
        damaged_mixed = add_yellowing(damaged_mixed, localized=random.choice([True, False]))
        damaged_mixed = add_scratches(damaged_mixed)
        save_pair(img, damaged_mixed, output_path, file_hash, "mixed", path.suffix)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='../datasets/processed')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    process_images_recursive(args.input, args.output)
