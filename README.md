# 🖼️ Image Restoration: Deterministic vs. Generative

A comprehensive image restoration platform comparing **Deterministic Reconstruction (SwinIR-Light)** against **Generative AI Restoration (Stable Diffusion + ControlNet)**. This project is designed to handle denoising, deblurring, and complex physical damage removal from historical photographs.

---

## 🏗️ Project Structure

- **`training/`**: PyTorch models, Diffusion pipelines, and Benchmarking scripts.
- **`demo/`**: Web application suite for side-by-side comparison.
    - **`client/`**: React (Vite) dashboard with 3-way comparison sliders.
    - **`server/`**: Node.js Express backend bridging to multiple Python inference engines.

---

## 🚀 Training & Inference

### 1. Dual Restoration Engines
- **SwinIR-Light**: Uses Window-based Multi-head Self-Attention for pixel-accurate reconstruction.
- **SD + ControlNet**: Uses generative inpainting conditioned on Canny edges and Tiles to "re-draw" missing sections.

### 2. High-Performance Benchmarking (Google Colab)
For deep training and heavy diffusion inference:
1. Run `./prepare_for_colab.sh` in the project root to create `restoration_training.zip`.
2. Upload to **Google Drive** and open `training/colab_training.ipynb` in Colab.
3. Compare both methods on the exact same dataset using our fixed `--seed 42`.

---

## 🛠️ Data Preparation Details
The pipeline uses **Synthetic Degradation** with localized randomization:
- **Localized Damage**: Random masks (rectangles, ellipses) target specific parts of the image.
- **Realistic Aging**: Yellowing, color fading, and simulated paper tears.
- **Reproducibility**: All benchmarking runs are seeded for exact replicability.

---

## 🐳 Docker Deployment (Comparison Suite)
Run the entire dual-method demo with a single command:
```bash
docker-compose up --build
```
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:5000

---

## ✨ Features
- [x] Dual Engine Architecture (Deterministic vs. Generative).
- [x] Localized damage simulation with fixed seeding.
- [x] Multi-method comparison slider (Before / SwinIR / AI).
- [x] Optimized for 4GB VRAM (Tiling, xFormers, LoRA).
- [x] Professional UI with Real-time Performance tracking.
