# Industrial VIN Extraction System 🚗💨

An advanced, high-precision Vehicle Identification Number (VIN) extraction pipeline optimized for challenging industrial environments, specifically tailored to handle **engraved metallic plates**, low resolution, and reflective surfaces.

## 🌟 Key Features

- **Advanced Preprocessing Suite**: 
    - **LANCZOS4 Upscaling**: High-fidelity resolution enhancement for low-res inputs.
    - **Engraved-Text Enhancements**: CLAHE-based contrast boosts and light dilation tuned for metallic plates.
- **Hybrid OCR & Merging**:
    - **PaddleOCR Detection**: Robust text region detection on challenging industrial scenes.
    - **EasyOCR CRNN Recognition**: Greedy CTC decoding with a strict industrial whitelist.
    - **Sliding-Window Merging**: Smart overlap-aware merging with duplicate-tail suppression for long IDs.
- **Flexible Input Modes**:
    - **Image Mode**: Manual ROI selection (with automatic fallback ROI when GUI is unavailable).
    - **Video Mode**: Multi-frame extraction with blur filtering and confidence-weighted fusion.

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV
- **Detection**: PaddleOCR
- **Recognition**: EasyOCR (CRNN + CTC)
- **Deep Learning Frameworks**: PaddlePaddle, PyTorch
- **Language**: Python 3.9+

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python installed, then set up your environment:

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage

Run the extractor on an image:

```powershell
python main.py path\to\your\vin_image.jpg --mode image
```

Run the extractor on a video (multi-frame fusion):

```powershell
python main.py path\to\your\video.mp4 --mode video
```

## 🏗️ Project Structure

```text
├── vin_extractor/
│   ├── preprocessing.py  # CV filters & variant generation
│   ├── ocr_engine.py     # PaddleOCR detection + EasyOCR recognition + merging
│   └── pipeline.py       # Orchestration, image & video pipelines
├── main.py               # Entry point CLI
└── requirements.txt      # Project dependencies
```
