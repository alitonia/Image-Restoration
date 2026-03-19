import torch
import cv2
import numpy as np
import os
import argparse
from glob import glob
from PIL import Image
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import create_model
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from peft import PeftModel

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_clean_dir', type=str, required=True)
    parser.add_argument('--test_damaged_dir', type=str, required=True)
    parser.add_argument('--swinir_weights', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results_test')
    parser.add_argument('--num_test', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load SwinIR
    print("⏳ Loading SwinIR weights...")
    swinir = create_model(upscale=1).to(device)
    swinir.load_state_dict(torch.load(args.swinir_weights, map_location=device))
    swinir.eval()

    ai_pipe = None
    canny_detector = None
    if args.lora_path:
        print("⏳ Loading Stable Diffusion Loss-Guided (ControlNet+Img2Img) Pipeline...")
        model_id = "runwayml/stable-diffusion-v1-5"
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
        ai_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id, 
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        ai_pipe.load_lora_weights(args.lora_path)
        canny_detector = CannyDetector()

    clean_files = sorted(glob(os.path.join(args.test_clean_dir, '*.*')))
    damaged_files = sorted(glob(os.path.join(args.test_damaged_dir, '*.*')))
    
    psnr_swinir = []
    psnr_sd = []
    
    print(f"🚀 Testing on {args.num_test} images...")
    for i in tqdm(range(min(args.num_test, len(clean_files)))):
        fname = os.path.basename(clean_files[i])
        clean_img = cv2.imread(clean_files[i])
        damaged_img = cv2.imread(damaged_files[i])
        
        # --- SWINIR INFERENCE ---
        h, w, _ = damaged_img.shape
        # Pad for SwinIR (needs multiple of 64)
        ph, pw = (h // 64 + 1) * 64, (w // 64 + 1) * 64
        padded = cv2.copyMakeBorder(damaged_img, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT)
        
        inp = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = swinir(inp)
            output = output.squeeze().cpu().numpy().transpose(1, 2, 0)
            output = (output * 255.0).clip(0, 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)[:h, :w, :]

        # Calculate SwinIR Metric
        psnr = calculate_psnr(clean_img, output)
        psnr_swinir.append(psnr)

        # --- AI INFERENCE (Simplified for speed) ---
        ai_result = np.zeros_like(clean_img)
        if ai_pipe:
            prompt = "a perfectly restored antique photo, high detail, sharp"
            
            # 1. Structure Skeleton (Filtered to ignore random noise/scratches via median blur)
            blurred_for_canny = cv2.medianBlur(damaged_img, 9)
            canny_raw = Image.fromarray(cv2.cvtColor(blurred_for_canny, cv2.COLOR_BGR2RGB)).resize((512, 512))
            control_image = canny_detector(canny_raw, low_threshold=100, high_threshold=200)
            
            # 2. Base colors and local detail layout (The actual damaged image)
            init_image = Image.fromarray(cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB)).resize((512, 512))
            
            with torch.autocast(device.type):
                ai_gen = ai_pipe(prompt=prompt, image=init_image, control_image=control_image, strength=0.7, controlnet_conditioning_scale=1.0, num_inference_steps=20).images[0]
            ai_gen = np.array(ai_gen.resize((w, h)))
            ai_result = cv2.cvtColor(ai_gen, cv2.COLOR_RGB2BGR)
            
            # Calculate SD-LoRA Metric
            psnr_sd_val = calculate_psnr(clean_img, ai_result)
            psnr_sd.append(psnr_sd_val)
            sd_psnr_text = f" | SD: {psnr_sd_val:.1f}dB"
        else:
            sd_psnr_text = ""

        # --- SAVE GRID [Clean | Damaged | SwinIR | AI] ---
        grid = np.hstack([clean_img, damaged_img, output, ai_result])
        cv2.putText(grid, f"SwinIR: {psnr:.1f}dB{sd_psnr_text}", (w * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(args.output_dir, f"test_{i}_{fname}"), grid)

    print(f"\n📊 Summary (SwinIR):")
    print(f"Average PSNR: {np.mean(psnr_swinir):.2f} dB")
    if len(psnr_sd) > 0:
        print(f"\n📊 Summary (SD-LoRA):")
        print(f"Average PSNR: {np.mean(psnr_sd):.2f} dB")
    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()
