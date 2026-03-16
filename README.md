# 🖼️ Image Restoration with Swin Transformer

A high-performance image restoration project using **SwinIR-Light**. This project is designed to handle denoising, deblurring, and removing physical damage (scratches, yellowing) from historical photographs.

---

## 🏗️ Project Structure

- **`training/`**: PyTorch implementation, training scripts, and utilities.
- **`demo/`**: Web application suite.
    - **`client/`**: React (Vite) frontend with a comparison slider.
    - **`server/`**: Node.js Express backend bridging to the Python model.

---

## 🚀 Training Guide

### 1. Local Setup (Validation)
If you want to test the pipeline locally on a small dataset:
```bash
cd training
# Create virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac
# Install dependencies
pip install -r requirements.txt

# Run Lite Data Preparation (Cleans old data and uses Seed 42)
./prepare_lite.sh

# Run quick training test
python scripts/train.py --clean_dir ./datasets/processed/clean --damaged_dir ./datasets/processed/damaged --epochs 1
```

### 2. High-Performance Training (Google Colab)
For deep training sessions on powerful GPUs:
1. Run `./prepare_for_colab.sh` in the project root to create `restoration_training.zip`.
2. Upload `restoration_training.zip` to the root of your **Google Drive**.
3. Open `training/colab_training.ipynb` in [Google Colab](https://colab.research.google.com).
4. Follow the notebook steps to mount Drive and start training.

---

## 🛠️ Data Preparation Details
The pipeline uses **Synthetic Degradation** with the following features:
- **Localized Damage**: Random masks (rectangles, ellipses) ensure damage only affects parts of the image.
- **Realistic Aging**: Yellowing, color fading, and simulated damping/water stains.
- **Deduplication**: SHA256 hashing prevents redundant processing and ensures unique mapping.
- **Reproducibility**: All scripts support a `--seed` argument for replicable results.

---
## 🐳 Docker Deployment (Demo Only)
If you have Docker and Docker Compose installed, you can run the entire demo suite with a single command:
```bash
docker-compose up --build
```
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:5000

---

## 🖥️ Running the Web Demo Manually

### Backend
```bash
cd demo/server
npm install
npm start
```

### Frontend
```bash
cd demo/client
npm install
npm run dev
```

---

## 📊 Evaluation
Run the evaluation script to calculate **PSNR** and **SSIM** metrics:
```bash
cd training
python scripts/evaluate.py --clean_dir ./datasets/processed/clean --damaged_dir ./datasets/processed/damaged --weights ../weights/swinir_restoration.pth
```

---

## ✨ Features
- [x] Window-based Multi-head Self-Attention (W-MSA).
- [x] Localized damage simulation (Noise, Blur, Scratches, Yellowing).
- [x] Optimized for low-resource hardware (AMP, Patch-based training).
- [x] Professional UI with Glassmorphism and comparison slider.
