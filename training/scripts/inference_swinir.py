import torch
import cv2
import numpy as np
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import create_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--weights', type=str, default=None, help='Model weights path')
    args = parser.parse_args()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(upscale=2) # Assuming 2x for SR
    
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
    
    model.to(device)
    model.eval()

    # Load and process image
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not load image {args.input}")
        return

    # Pre-process
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0)
    img = img.to(device)

    # Inference
    with torch.no_grad():
        # Padding to makesure it's divisible by window_size (8)
        window_size = 8
        b, c, h, w = img.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        output = model(img)
        
        # Crop back if needed (SwinIR handles this internally but just in case for upscale)
        _, _, oh, ow = output.shape
        output = output[:, :, :oh - pad_h*2, :ow - pad_w*2]

    # Post-process
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    # Save
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    cv2.imwrite(args.output, output)
    print(f"Successfully saved restored image to {args.output}")

if __name__ == '__main__':
    main()
