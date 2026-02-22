import os

# Region of Interest (ROI) configuration
# Format: [ymin, xmin, ymax, xmax] as fractions of image size (0.0 to 1.0)
# Defaulting to middle section of the image
DEFAULT_ROI = {
    "ymin": 0.3,
    "xmin": 0.1,
    "ymax": 0.7,
    "xmax": 0.9
}

# OCR Configuration
PADLLE_OCR_LANG = 'en'
# Minimum confidence to accept a result. Lower (e.g. 0.35) helps with difficult engravings.
RECOGNITION_CONFIDENCE_THRESHOLD = 0.40
# Fallback: if no result meets threshold, accept best result above this (and mark low confidence).
RECOGNITION_CONFIDENCE_FALLBACK = 0.25
# Last resort: accept best result with any confidence (e.g. for very difficult engravings).
RECOGNITION_CONFIDENCE_ANY = 0.01



# Blur Detection Threshold
# Laplacian variance below this value is considered too blurry
BLUR_THRESHOLD = 30.0

# General ID Validation (No longer restricted to VIN)
ID_REGEX = r"^[A-Z0-9\-_./: ]+$"

