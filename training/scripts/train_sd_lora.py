import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    DDPMScheduler, 
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from glob import glob

class SDLoraDataset(Dataset):
    def __init__(self, image_dir, caption="a restored antique photo", tokenizer=None, size=512):
        self.image_files = sorted(glob(os.path.join(image_dir, '*.*')))
        if len(self.image_files) == 0:
            raise ValueError(f"No images found for SD-LoRA training in {image_dir}!")
        print(f"SD-LoRA Dataset: Found {len(self.image_files)} training images.")
        self.caption = caption
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # Pixel values to [-1, 1]
        import numpy as np
        pixel_values = (np.array(image).astype(np.float32) / 127.5) - 1.0
        pixel_values = torch.from_numpy(pixel_values).permute(2, 0, 1)

        # Tokenize caption
        inputs = self.tokenizer(
            self.caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return {"pixel_values": pixel_values, "input_ids": inputs.input_ids[0]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to clean/restored images')
    parser.add_argument('--model_id', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--output_dir', type=str, default="../weights/sd_lora_antique")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--validation_prompt', type=str, default="a restored antique photo", help='Prompt for validation')
    parser.add_argument('--num_validation_images', type=int, default=1, help='How many validation images per epoch')
    parser.add_argument('--val_dir', type=str, default=None, help='Directory to save validation images')
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="fp16")
    set_seed(args.seed)

    # 1. Load basic components
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae = StableDiffusionPipeline.from_pretrained(args.model_id).vae
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")

    # 2. Freeze VAE and Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 3. Add LoRA to UNet
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 4. Optimizer & Dataset
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    dataset = SDLoraDataset(args.image_dir, tokenizer=tokenizer, size=args.resolution)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 5. Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # 6. Prepare for acceleration
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # 7. Resume logic
    start_epoch = 0
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint_latest")
    if args.resume and os.path.exists(checkpoint_dir):
        accelerator.print(f"Loading accelerator state from {checkpoint_dir}")
        accelerator.load_state(checkpoint_dir)
        # We need a way to track the epoch. Let's look for epoch folders.
        epochs = [int(d.split('_')[1]) for d in os.listdir(args.output_dir) if d.startswith('epoch_')]
        if epochs:
            start_epoch = max(epochs) + 1
            accelerator.print(f"Resuming from epoch {start_epoch}")

    # 8. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.detach().item()})

        # Save checkpoint and accelerator state
        if accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            unet.save_pretrained(save_path)
            # Save as latest for easy loading in test scripts
            unet.save_pretrained(os.path.join(args.output_dir, "latest"))
            
            # Save latest state for resuming
            accelerator.save_state(checkpoint_dir)
            accelerator.print(f"✅ Checkpoint saved at {save_path}")

            # Visual Validation
            if args.val_dir:
                accelerator.print(f"🎨 Generating validation images for epoch {epoch}...")
                unet.eval()
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.model_id, 
                    unet=accelerator.unwrap_model(unet), 
                    text_encoder=text_encoder, 
                    vae=vae, 
                    torch_dtype=weight_dtype
                ).to(accelerator.device)
                
                os.makedirs(args.val_dir, exist_ok=True)
                for i in range(args.num_validation_images):
                    image = pipeline(args.validation_prompt).images[0]
                    image.save(os.path.join(args.val_dir, f"val_{epoch}_{i}.png"))
                
                del pipeline # Free memory
                torch.cuda.empty_cache()
                unet.train()

    print("LoRA training complete!")

if __name__ == "__main__":
    main()
