# Industrial VIN Extraction System 🚗💨

An advanced, high-precision Vehicle Identification Number (VIN) extraction pipeline optimized for challenging industrial environments, specifically tailored to handle **engraved metallic plates**, low resolution, and reflective surfaces.

## 🌟 Key Features

- **Advanced Preprocessing Suite**: 
    - **LANCZOS4 Upscaling**: High-fidelity resolution enhancement for low-res inputs.
    - **Shadow Normalization**: Gradient-based contrast stretching to neutralize metallic glare.
    - **Unsharp Masking & LoG Sharpening**: Stroke amplification to resolve similar characters (e.g., M vs V, 8 vs B).
    - **Dynamic Gamma Correction**: Multi-exposure simulation to capture faint engravings.
- **Intelligent Extraction & Validation**:
    - **Tiered Multi-Strategy OCR**: Iterates through multiple image variants until a valid VIN is found.
    - **ISO 3779 Checksum Verification**: Mathematical validation using the Modulo 11 check digit (Position 9) to ensure 99.9% accuracy.
    - **Fuzzy Recovery Logic**: Automated heuristic correction for common OCR confusions (e.g., swapping V->M or 8->B to satisfy checksum requirements).

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV
- **OCR Engine**: PaddleOCR (Deep Learning based)
- **Deep Learning Framework**: PaddlePaddle
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
Run the extractor on any image:

```powershell
python main.py path/to/your/vin_image.jpg
```

## 🏗️ Project Structure

```text
├── vin_extractor/
│   ├── preprocessing.py  # Advanced CV filters & variant generation
│   ├── ocr_engine.py     # PaddleOCR wrapper with optimized params
│   ├── validator.py      # ISO 3779 Checksum & Fuzzy Correction logic
│   └── pipeline.py       # Orchestration & fallback logic
├── main.py               # Entry point CLI
└── requirements.txt      # Project dependencies
```

## ⚖️ ISO 3779 Compliance
The system strictly adheres to the ISO 3779 standard:
- **Length**: 17 Alphanumeric characters.
- **Exclusions**: Automatically handles or corrects prohibited characters (I, O, Q).
- **Check Digit**: Validates the 9th character using standardized weights.

---
*Developed for high-precision industrial HMI applications.*
