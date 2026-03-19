import torch
import cv2
import numpy as np
import argparse
import os
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--prompt', type=str, default="high quality, professional restoration, antique photo", help='Text prompt')
    parser.add_argument('--negative_prompt', type=str, default="bad quality, blurry, distorted, changed identity", help='Negative prompt')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to custom LoRA weights')
    args = parser.parse_args()

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load ControlNet (Canny for structural preservation)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # 2. Load Pipeline
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=controlnet,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # 3. Load custom LoRA if provided
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading custom PEFT LoRA from {args.lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)

    # 4. Memory Optimizations for 4GB VRAM
    if device == "cuda":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling() # Good for higher resolution output on low VRAM
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("xformers not available, skipping...")

    # 5. Prepare Control Image (Canny Edge)
    input_image = Image.open(args.input).convert("RGB")
    # Resize to something manageable but keeping aspect ratio
    input_image.thumbnail((512, 512))
    
    # Structure Skeleton Filter (blur to hide scratches from the geometric control net map)
    cv_img = np.array(input_image)
    cv_img_bgr = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    blurred_for_canny = cv2.medianBlur(cv_img_bgr, 7)
    canny_raw = Image.fromarray(cv2.cvtColor(blurred_for_canny, cv2.COLOR_BGR2RGB))
    
    canny_detector = CannyDetector()
    control_image = canny_detector(canny_raw, low_threshold=100, high_threshold=200)

    # 6. Inference
    generator = torch.manual_seed(42)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type):
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=input_image,
            control_image=control_image,
            strength=0.7,
            num_inference_steps=30, # Increased steps for quality
            generator=generator,
            controlnet_conditioning_scale=1.0,
        ).images[0]

    # 6. Save
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    result.save(args.output)
    print(f"Successfully saved SD+ControlNet restored image to {args.output}")

if __name__ == "__main__":
    main()
