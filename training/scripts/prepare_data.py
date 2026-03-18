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
    mask_type = random.choice(['full', 'rect', 'ellipse', 'multiply', 'brush', 'poly'])
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
        axes = (random.randint(20, w//2), random.randint(20, h//2))
        angle = random.randint(0, 360)
        cv2.ellipse(sub_mask, center, axes, angle, 0, 360, 1.0, -1)
        return sub_mask

    if mask_type == 'multiply':
        # Multiple small patches
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        for _ in range(random.randint(3, 8)):
              center = (random.randint(0, w), random.randint(0, h))
              axes = (random.randint(5, w//6), random.randint(5, h//6))
              cv2.ellipse(sub_mask, center, axes, random.randint(0, 360), 0, 360, 1.0, -1)
        return sub_mask
    
    if mask_type == 'brush':
        # Simulate a brush stroke / tear
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        start_point = (random.randint(0, w), random.randint(0, h))
        for _ in range(random.randint(5, 15)):
            end_point = (start_point[0] + random.randint(-w//4, w//4), start_point[1] + random.randint(-h//4, h//4))
            cv2.line(sub_mask, start_point, end_point, 1.0, random.randint(10, 40))
            start_point = end_point
        return sub_mask

    if mask_type == 'poly':
        # Random polygon for sharp tears
        sub_mask = np.zeros((h, w, 1), dtype=np.float32)
        num_pts = random.randint(3, 6)
        pts = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(num_pts)])
        cv2.fillPoly(sub_mask, [pts], 1.0)
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

def add_folds(img):
    """Simulate paper fold lines."""
    draw_img = img.copy()
    num_folds = random.randint(1, 4)
    for _ in range(num_folds):
        if random.random() > 0.5: # Vertical-ish
            x = random.randint(0, img.shape[1])
            cv2.line(draw_img, (x, 0), (x + random.randint(-10, 10), img.shape[0]), (0.8, 0.8, 0.8), random.randint(1, 2))
        else: # Horizontal-ish
            y = random.randint(0, img.shape[0])
            cv2.line(draw_img, (0, y), (img.shape[1], y + random.randint(-10, 10)), (0.8, 0.8, 0.8), random.randint(1, 2))
    return draw_img

def add_water_stains(img):
    """Simulate water or coffee stains."""
    h, w, _ = img.shape
    num_stains = random.randint(1, 3)
    for _ in range(num_stains):
        stain_layer = np.zeros((h, w, 1), dtype=np.float32)
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(20, w//3), random.randint(20, h//3))
        # Brownish stain for ancient look
        cv2.ellipse(stain_layer, center, axes, random.randint(0, 360), 0, 360, random.uniform(0.1, 0.4), -1)
        stain_color = np.array([0.4, 0.6, 0.8]) # BGR: Low Blue, High Red
        img = img * (1 - stain_layer) + (img * stain_color) * stain_layer
    return np.clip(img, 0, 1)

def add_mold_spots(img):
    """Simulate dark mold or foxing spots."""
    h, w, _ = img.shape
    for _ in range(random.randint(30, 80)):
        center = (random.randint(0, w), random.randint(0, h))
        radius = random.randint(1, 3)
        # Darker spots
        color = (random.uniform(0.05, 0.2), random.uniform(0.05, 0.2), random.uniform(0.05, 0.2))
        cv2.circle(img, center, radius, color, -1)
    return img

def add_noise_patch(img):
    """Replaces a part of the image with pure random noise."""
    h, w, c = img.shape
    mask = get_random_mask(h, w) # Use the existing mask generator
    noise_patch = np.random.uniform(0, 1, (h, w, c)).astype(np.float32)
    img = img * (1 - mask) + noise_patch * mask
    return np.clip(img, 0, 1)

def add_compression(img):
    """Simulate JPEG compression artifacts."""
    # Convert back to uint8 for cv2.imencode
    img_uint8 = (img * 255).astype(np.uint8)
    quality = random.randint(10, 50)
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

def process_images_recursive(input_dir, output_dir, multiplier=1, split_ratio=0.9):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.rglob(ext))
    
    # Shuffle for random split
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * split_ratio)
    train_paths = image_paths[:split_idx]
    test_paths = image_paths[split_idx:]
    
    print(f"Found {len(image_paths)} images. Split: {len(train_paths)} Train / {len(test_paths)} Test. Multiplier: {multiplier}.")
    
    processed_hashes = set()
    
    # Sub-task lists for easier looping
    tasks = [('train', train_paths), ('test', test_paths)]
    
    for subset_name, paths in tasks:
        subset_output = output_path / subset_name
        print(f"Processing {subset_name} set...")
        
        for path in tqdm(paths):
            file_hash = get_file_hash(path)
            if file_hash in processed_hashes: continue
            processed_hashes.add(file_hash)

            img = cv2.imread(str(path))
            if img is None: continue
            
            # Resize slightly if too large for memory
            if img.shape[0] > 1024 or img.shape[1] > 1024:
                 img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

            img_float = img.astype(np.float32) / 255.0
            
            for m in range(multiplier):
                # Apply a sequence of random damages (Multiple types per image)
                damaged = img_float.copy()
                
                # Damage List: [Probability, Function]
                damage_pipeline = [
                    (0.8, lambda i: add_blur(i, localized=random.choice([True, False]))),
                    (0.8, lambda i: add_noise(i, localized=random.choice([True, False]))),
                    (0.6, lambda i: add_yellowing(i, localized=random.choice([True, False]))),
                    (0.7, add_scratches),
                    (0.4, add_folds),
                    (0.4, add_water_stains),
                    (0.5, add_mold_spots),
                    (0.3, add_noise_patch),
                    (0.6, add_compression),
                ]
                
                # Randomly shuffle and apply a subset
                random.shuffle(damage_pipeline)
                applied_count = 0
                for prob, func in damage_pipeline:
                    if random.random() < prob:
                        damaged = func(damaged)
                        applied_count += 1
                    if applied_count >= 5: break # Max 5 types per image
                
                # Ensure at least one damage is applied
                if applied_count == 0:
                    damaged = add_noise(damaged)
                
                suffix = f"complex_{m}" if multiplier > 1 else "complex"
                save_pair(img, damaged, subset_output, file_hash, suffix, path.suffix)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='../datasets/processed')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--multiplier', type=int, default=1, help='How many damaged images per source')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='Train/Test split ratio')
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    process_images_recursive(args.input, args.output, args.multiplier, args.split_ratio)
