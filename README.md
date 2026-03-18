# 🖼️ Image Restoration: Deterministic vs. Generative

A comprehensive image restoration platform comparing **Deterministic Reconstruction (SwinIR-Light)** against **Generative AI Restoration (Stable Diffusion + ControlNet)**. This project handles denoising, deblurring, and complex physical damage removal (tears, folds, mold) from historical photographs.

---

## 🏗️ Project Structure

- **`training/`**: PyTorch models, Diffusion pipelines, and Benchmarking scripts.
    - **`scripts/prepare_data.py`**: Advanced synthetic degradation engine.
    - **`scripts/train_swinir.py`**: Resumable SwinIR-Light training.
    - **`scripts/train_sd_lora.py`**: Resumable Stable Diffusion LoRA training.
- **`demo/`**: Web application suite for side-by-side comparison.
    - **`client/`**: React (Vite) dashboard with 3-way comparison sliders and real-time tracking.
    - **`server/`**: Node.js Express backend bridging to multiple Python inference engines.

---

## 🚀 Training & Inference

### 1. Dual Restoration Engines
- **SwinIR-Light**: Uses Window-based Multi-head Self-Attention for pixel-accurate reconstruction.
- **SD + ControlNet**: Uses generative inpainting conditioned on Canny edges. Now supports **Custom LoRA** for specialized antique textures.

### 2. Progressive Training (Resumable)
Both training tracks now support automatic resuming if a session is interrupted (ideal for Google Colab):
- **SwinIR**: Uses `--resume` to load state-dict snapshots.
- **SD LoRA**: Uses `accelerate.save_state()` for full-context restoration.

### 3. High-Performance Benchmarking (Google Colab)
1. Run `./prepare_for_colab.sh` to package scripts.
2. Upload `restoration_training.zip` to Google Drive.
3. Open `training/colab_training.ipynb` for GPU-accelerated training with **automatic Drive checkpointing**.

---

## 🛠️ Data Synthesis Engine
The pipeline uses **Complex Stochastic Degradation** with multiple simultaneous damage types:
- **Localized Damage**: Irregular shapes (Brush, Poly, Ellipse) for noise patches and tears.
- **Physical Aging**: Scratches, paper folds, water/coffee stains, and mold spots (foxing).
- **Modes**: 
    - `./prepare_lite.sh`: 10x multiplier for testing on small datasets.
    - `./prepare_full.sh`: 1x multiplier for high-efficiency large dataset processing.
- **Determinism**: All synthesis and training runs use global seeding (default: `--seed 42`).

---

## 🐳 Docker Deployment
Run the entire dual-method demo locally:
```bash
docker-compose up --build
```
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:5000

---

## ✨ Key Features
- [x] **Dual Engine Architecture** (Deterministic vs. Generative).
- [x] **Complex Damage Simulation** (Tears, Folds, Stains, Mold).
- [x] **Progressive Training** (Automatic resume/checkpointing).
- [x] **LoRA Inference Support** in `inference_sd.py`.
- [x] **Optimized for 4GB VRAM** (Tiling, xFormers, VAE Tiling).
- [x] **Fully Deterministic** results via global seeding.
