import torch
import cv2
import os
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import create_model

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--damaged_dir', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(upscale=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    clean_files = sorted(glob(os.path.join(args.clean_dir, '*.*')))
    damaged_files = sorted(glob(os.path.join(args.damaged_dir, '*.*')))

    psnr_total = 0
    ssim_total = 0
    count = 0

    print("Evaluating model...")
    for c_file, d_file in tqdm(zip(clean_files, damaged_files), total=len(clean_files)):
        clean_img = cv2.imread(c_file)
        damaged_img = cv2.imread(d_file)

        # Preprocess
        img_input = damaged_img.astype(np.float32) / 255.0
        img_input = torch.from_numpy(np.transpose(img_input[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_input)

        # Postprocess
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)

        # Calculate metrics
        psnr = psnr_metric(clean_img, output)
        ssim = ssim_metric(clean_img, output, channel_axis=2)

        psnr_total += psnr
        ssim_total += ssim
        count += 1

    print(f"\nFinal Results:")
    print(f"Average PSNR: {psnr_total / count:.2f} dB")
    print(f"Average SSIM: {ssim_total / count:.4f}")

if __name__ == "__main__":
    evaluate()
