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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_damaged_dir', type=str, required=True)
    parser.add_argument('--swinir_weights', type=str, required=True)
    parser.add_argument('--lora_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results_recursive')
    parser.add_argument('--num_test', type=int, default=2)
    parser.add_argument('--passes', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load SwinIR
    print("⏳ Loading SwinIR...")
    swinir = create_model(upscale=1).to(device)
    swinir.load_state_dict(torch.load(args.swinir_weights, map_location=device))
    swinir.eval()

    # 2. Load AI
    ai_pipe = None
    canny_detector = None
    if args.lora_path:
        print("⏳ Loading SD-LoRA...")
        model_id = "runwayml/stable-diffusion-v1-5"
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to(device)
        ai_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id, 
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        ai_pipe.unet = PeftModel.from_pretrained(ai_pipe.unet, args.lora_path)
        canny_detector = CannyDetector()

    damaged_files = sorted(glob(os.path.join(args.test_damaged_dir, '*.*')))
    
    print(f"🚀 Running recursive test with n={args.passes} passes...")
    for i in range(min(args.num_test, len(damaged_files))):
        fname = os.path.basename(damaged_files[i])
        
        # Initial damaged images
        original_img = cv2.imread(damaged_files[i])
        h, w, _ = original_img.shape
        
        curr_swinir = original_img.copy()
        curr_sd = original_img.copy()
        
        swinir_history = [original_img]
        sd_history = [original_img]
        
        print(f"\\n🖼️ Processing image {fname}...")
        for p in tqdm(range(args.passes)):
            # --- SWINIR INFERENCE ---
            ph, pw = (h // 64 + 1) * 64, (w // 64 + 1) * 64
            padded = cv2.copyMakeBorder(curr_swinir, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT)
            inp = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = torch.from_numpy(np.transpose(inp, (2, 0, 1))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = swinir(inp).squeeze().cpu().numpy().transpose(1, 2, 0)
                out = (out * 255.0).clip(0, 255).astype(np.uint8)
                curr_swinir = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)[:h, :w, :]
            
            # Put label on it
            swinir_disp = curr_swinir.copy()
            cv2.putText(swinir_disp, f"Pass {p+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            swinir_history.append(swinir_disp)
            
            # --- SD-LoRA INFERENCE ---
            if ai_pipe:
                prompt = "a perfectly restored antique photo, high detail, sharp"
                blurred = cv2.medianBlur(curr_sd, 9)
                canny_raw = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)).resize((512, 512))
                control_image = canny_detector(canny_raw, low_threshold=100, high_threshold=200)
                init_image = Image.fromarray(cv2.cvtColor(curr_sd, cv2.COLOR_BGR2RGB)).resize((512, 512))
                
                with torch.autocast(device.type):
                    ai_gen = ai_pipe(prompt=prompt, image=init_image, control_image=control_image, strength=0.7, controlnet_conditioning_scale=1.0, num_inference_steps=20).images[0]
                ai_gen = np.array(ai_gen.resize((w, h)))
                curr_sd = cv2.cvtColor(ai_gen, cv2.COLOR_RGB2BGR)
                
                sd_disp = curr_sd.copy()
                cv2.putText(sd_disp, f"Pass {p+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                sd_history.append(sd_disp)

        # Draw original text on first image
        cv2.putText(swinir_history[0], "Original Damaged", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if ai_pipe:
            cv2.putText(sd_history[0], "Original Damaged", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save SWINIR GIF & grid
        swinir_pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in swinir_history]
        swinir_pil[0].save(os.path.join(args.output_dir, f"{i}_swinir_effect.gif"), save_all=True, append_images=swinir_pil[1:], duration=600, loop=0)
        cv2.imwrite(os.path.join(args.output_dir, f"{i}_swinir_grid.png"), np.hstack(swinir_history))

        # Save SD GIF & grid
        if ai_pipe:
            sd_pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in sd_history]
            sd_pil[0].save(os.path.join(args.output_dir, f"{i}_sd_effect.gif"), save_all=True, append_images=sd_pil[1:], duration=600, loop=0)
            cv2.imwrite(os.path.join(args.output_dir, f"{i}_sd_grid.png"), np.hstack(sd_history))
            
    print(f"\\n✅ Saved grids and animations to {args.output_dir}")

if __name__ == "__main__":
    main()
