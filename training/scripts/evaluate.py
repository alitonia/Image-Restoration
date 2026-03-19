import torch
import cv2
import os
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import normalized_root_mse as nrmse_metric
from skimage.metrics import mean_squared_error as mse_metric
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model import create_model

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean_dir', type=str, required=True, help='Ground truth images')
    parser.add_argument('--damaged_dir', type=str, required=False, help='Damaged images (only if using --weights)')
    parser.add_argument('--restored_dir', type=str, required=False, help='Folder of already restored images for evaluation')
    parser.add_argument('--weights', type=str, required=False, help='SwinIR model weights')
    parser.add_argument('--lora_path', type=str, required=False, help='Path to SD-LoRA weights to evaluate')
    args = parser.parse_args()

    clean_files = sorted(glob(os.path.join(args.clean_dir, '*.*')))
    
    psnr_total = 0
    ssim_total = 0
    mse_total = 0
    nrmse_total = 0
    count = 0

    if args.weights:
        print(f"Evaluating SwinIR model with weights: {args.weights}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(upscale=1).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device))
        model.eval()
        
        damaged_files = sorted(glob(os.path.join(args.damaged_dir, '*.*')))
        
        for c_file, d_file in tqdm(zip(clean_files, damaged_files), total=len(clean_files)):
            clean_img = cv2.imread(c_file)
            damaged_img = cv2.imread(d_file)
            
            img_input = damaged_img.astype(np.float32) / 255.0
            img_input = torch.from_numpy(np.transpose(img_input[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_input)
            
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            
            psnr = psnr_metric(clean_img, output)
            ssim = ssim_metric(clean_img, output, channel_axis=2)
            mse = mse_metric(clean_img, output)
            nrmse = nrmse_metric(clean_img, output)
            
            psnr_total += psnr
            ssim_total += ssim
            mse_total += mse
            nrmse_total += nrmse
            count += 1

    elif args.restored_dir:
        print(f"Evaluating restored folder: {args.restored_dir}")
        restored_files = sorted(glob(os.path.join(args.restored_dir, '*.*')))
        
        for c_file, r_file in tqdm(zip(clean_files, restored_files), total=len(clean_files)):
            clean_img = cv2.imread(c_file)
            restored_img = cv2.imread(r_file)
            
            # Resize restored if needed to match GT (SD might produce 512x512)
            if restored_img.shape != clean_img.shape:
                restored_img = cv2.resize(restored_img, (clean_img.shape[1], clean_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            psnr = psnr_metric(clean_img, restored_img)
            ssim = ssim_metric(clean_img, restored_img, channel_axis=2)
            mse = mse_metric(clean_img, restored_img)
            nrmse = nrmse_metric(clean_img, restored_img)
            
            psnr_total += psnr
            ssim_total += ssim
            mse_total += mse
            nrmse_total += nrmse
            count += 1
            
    elif args.lora_path:
        print(f"Evaluating SD-LoRA model on the whole dataset (this may take a few minutes)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
        from controlnet_aux import CannyDetector
        from PIL import Image
        from peft import PeftModel
        
        model_id = "runwayml/stable-diffusion-v1-5"
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id, 
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)
        pipe.set_progress_bar_config(disable=True)
        canny_detector = CannyDetector()
        
        damaged_files = sorted(glob(os.path.join(args.damaged_dir, '*.*')))
        prompt = "a perfectly restored antique photo, high detail, sharp"
        
        for c_file, d_file in tqdm(zip(clean_files, damaged_files), total=len(clean_files)):
            clean_img = cv2.imread(c_file)
            damaged_img = cv2.imread(d_file)
            h, w = clean_img.shape[:2]
            
            # 1. Structure Skeleton (Filtered to ignore random noise/scratches via median blur)
            blurred_for_canny = cv2.medianBlur(damaged_img, 9)
            canny_raw = Image.fromarray(cv2.cvtColor(blurred_for_canny, cv2.COLOR_BGR2RGB)).resize((512, 512))
            control_image = canny_detector(canny_raw, low_threshold=100, high_threshold=200)
            
            # 2. Base colors and local detail layout (The actual damaged image)
            init_image = Image.fromarray(cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB)).resize((512, 512))
            
            with torch.autocast(device.type):
                ai_gen = pipe(prompt=prompt, image=init_image, control_image=control_image, strength=0.7, controlnet_conditioning_scale=1.0, num_inference_steps=20).images[0]
                
            ai_gen = np.array(ai_gen.resize((w, h)))
            restored_img = cv2.cvtColor(ai_gen, cv2.COLOR_RGB2BGR)

            psnr = psnr_metric(clean_img, restored_img)
            ssim = ssim_metric(clean_img, restored_img, channel_axis=2)
            mse = mse_metric(clean_img, restored_img)
            nrmse = nrmse_metric(clean_img, restored_img)
            
            psnr_total += psnr
            ssim_total += ssim
            mse_total += mse
            nrmse_total += nrmse
            count += 1
    else:
        print("Error: Provide either --weights, --lora_path, or --restored_dir")
        return

    if count == 0:
        print("\nError: No images were evaluated. Please check if your dataset directories contain images and the paths are correct.")
        return

    print(f"\nFinal Results:")
    print(f"Average PSNR:  {psnr_total / count:.2f} dB")
    print(f"Average SSIM:  {ssim_total / count:.4f}")
    print(f"Average MSE:   {mse_total / count:.2f}")
    print(f"Average NRMSE: {nrmse_total / count:.4f}")

if __name__ == "__main__":
    evaluate()
