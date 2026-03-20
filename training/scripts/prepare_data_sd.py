import cv2
import numpy as np
import os
import random
import hashlib
from tqdm import tqdm
from pathlib import Path

def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_random_mask(h, w, mask_type=None):
    if not mask_type:
        mask_type = random.choice(['full', 'rect', 'ellipse', 'multiply', 'blob'])
    
    mask = np.ones((h, w, 1), dtype=np.float32)
    if mask_type == 'full': return mask
    
    if mask_type == 'rect':
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(x1 + 50, w), random.randint(y1 + 50, h)
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        sub_mask[y1:y2, x1:x2, :] = 1.0
        return sub_mask
        
    if mask_type == 'ellipse':
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(40, w//2), random.randint(40, h//2))
        angle = random.randint(0, 360)
        cv2.ellipse(sub_mask, center, axes, angle, 0, 360, 1.0, -1)
        return sub_mask

    if mask_type == 'multiply':
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(2, 5)):
              center = (random.randint(0, w), random.randint(0, h))
              axes = (random.randint(20, w//4), random.randint(20, h//4))
              cv2.ellipse(sub_mask, center, axes, random.randint(0, 360), 0, 360, 1.0, -1)
        return sub_mask
        
    if mask_type == 'blob':
        # Soft blurred blob
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(1, 3)):
            center = (random.randint(0, w), random.randint(0, h))
            axes = (random.randint(30, w//3), random.randint(30, h//3))
            cv2.ellipse(sub_mask, center, axes, random.randint(0, 360), 0, 360, 1.0, -1)
        sub_mask = cv2.GaussianBlur(sub_mask, (51, 51), 0)
        if len(sub_mask.shape) == 2:
            sub_mask = np.expand_dims(sub_mask, axis=-1)
        return sub_mask
    
    return mask

def add_heavy_blur(img):
    """Heavy blur destroys high-frequency textures (hair, pores) but keeps structure."""
    h, w, _ = img.shape
    kernel_size = random.choice([7, 9, 11, 15])
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    if random.random() > 0.5:
        # Localized blur
        mask = get_random_mask(h, w, 'blob')
        img_blur = blurred * mask + img * (1 - mask)
    else:
        img_blur = blurred
    return np.clip(img_blur, 0, 1)

def add_sepia_and_fading(img):
    """Destroys original colors, forcing SD to hallucinate/colorize them."""
    img_color = img.copy().astype(np.float32)
    
    # 1. Desaturate heavily (move towards grayscale)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_color = img_color * 0.2 + gray * 0.8
    
    # 2. Add Sepia/Yellowing tone
    # BGR format: 0=Blue, 1=Green, 2=Red
    img_color[:, :, 0] *= random.uniform(0.3, 0.6)  # Kill Blue
    img_color[:, :, 1] *= random.uniform(0.8, 1.0)  # Moderate Green
    img_color[:, :, 2] *= random.uniform(1.0, 1.3)  # Boost Red
    
    # 3. Fade contrast
    fade = random.uniform(0.4, 0.8)
    img_color = img_color * fade + (1 - fade) * 0.5 
    
    return np.clip(img_color, 0, 1)

def add_large_water_stains(img):
    """Large stains that obscure textures but allow Canny to easily find boundaries."""
    h, w, _ = img.shape
    num_stains = random.randint(1, 4)
    for _ in range(num_stains):
        stain_layer = np.zeros((h, w, 1), dtype=np.float32)
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(40, w//2), random.randint(40, h//2))
        cv2.ellipse(stain_layer, center, axes, random.randint(0, 360), 0, 360, random.uniform(0.3, 0.8), -1)
        
        # Blur the stain edges so Canny doesn't misinterpret the stain edge as a physical object
        stain_layer = cv2.GaussianBlur(stain_layer, (31, 31), 0)
        if len(stain_layer.shape) == 2:
            stain_layer = np.expand_dims(stain_layer, axis=-1)
        
        stain_color = np.array([0.2, 0.3, 0.5]) # BGR: Dark brown/rust
        img = img * (1 - stain_layer) + (img * stain_color) * stain_layer
    return np.clip(img, 0, 1)

def add_missing_patches(img):
    """Completely white/black missing regions (holes/tears) which SD excels at inpainting."""
    h, w, _ = img.shape
    num_holes = random.randint(1, 3)
    for _ in range(num_holes):
        mask = get_random_mask(h, w, 'multiply')
        fill_color = random.choice([0.0, 1.0]) # Pure black or pure white holes
        img = img * (1 - mask) + fill_color * mask
    return np.clip(img, 0, 1)

def add_heavy_compression(img):
    """Destroy fine details with aggressive JPEG artifacts."""
    img_uint8 = (img * 255).astype(np.uint8)
    quality = random.randint(5, 20) # Very low quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg.astype(np.float32) / 255.0

def save_pair(clean_img_uint8, damaged_img_float, output_path, name_prefix, suffix, file_ext):
    clean_dir = output_path / 'clean'
    damaged_dir = output_path / 'damaged'
    clean_dir.mkdir(parents=True, exist_ok=True)
    damaged_dir.mkdir(parents=True, exist_ok=True)
    
    target_name = f"{name_prefix}_{suffix}{file_ext}"
    cv2.imwrite(str(clean_dir / target_name), clean_img_uint8)
    damaged_img_uint8 = (damaged_img_float * 255).astype(np.uint8)
    cv2.imwrite(str(damaged_dir / target_name), damaged_img_uint8)

def process_images_for_sd(input_dir, output_dir, multiplier=1, split_ratio=0.9, limit=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.rglob(ext))
    
    # Sort to ensure consistent "first 10" if seed isn't provided
    image_paths = sorted(image_paths)
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]
        
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * split_ratio)
    train_paths = image_paths[:split_idx]
    test_paths = image_paths[split_idx:]
    
    print(f"Found {len(image_paths)} images. Split: {len(train_paths)} Train / {len(test_paths)} Test. Multiplier: {multiplier}.")
    
    processed_hashes = set()
    tasks = [('train', train_paths), ('test', test_paths)]
    
    for subset_name, paths in tasks:
        subset_output = output_path / subset_name
        print(f"Processing {subset_name} set for SD Optimization...")
        
        for path in tqdm(paths):
            file_hash = get_file_hash(path)
            if file_hash in processed_hashes: continue
            processed_hashes.add(file_hash)

            img = cv2.imread(str(path))
            if img is None: continue
            
            if img.shape[0] > 1024 or img.shape[1] > 1024:
                 img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

            img_float = img.astype(np.float32) / 255.0
            
            for m in range(multiplier):
                damaged = img_float.copy()
                
                # SD-Optimized Damage Pipeline
                # Focus: Destroy texture and color, keep underlying global structure
                damage_pipeline = [
                    (0.9, add_heavy_blur),
                    (0.9, add_sepia_and_fading),
                    (0.6, add_large_water_stains),
                    (0.4, add_missing_patches),
                    (0.8, add_heavy_compression),
                ]
                
                random.shuffle(damage_pipeline)
                applied_count = 0
                for prob, func in damage_pipeline:
                    if random.random() < prob:
                        damaged = func(damaged)
                        applied_count += 1
                    if applied_count >= 3: break
                
                # Always compress at least a little bit at the end to unify the image
                if random.random() < 0.5:
                    damaged = add_heavy_compression(damaged)
                
                suffix = f"sd_optimized_{m}" if multiplier > 1 else "sd_optimized"
                save_pair(img, damaged, subset_output, file_hash, suffix, path.suffix)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='../datasets/processed_sd')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--multiplier', type=int, default=1)
    parser.add_argument('--split_ratio', type=float, default=0.9)
    parser.add_argument('--limit', type=int, default=None, help='Limit processing to first N images')
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    process_images_for_sd(args.input, args.output, args.multiplier, args.split_ratio, args.limit)
