import cv2
import numpy as np
import logging
import re
from .preprocessing import PreProcessor
from .ocr_engine import OCREngine
# VINValidator removed - no longer using VIN-specific validation
from .config import (
    DEFAULT_ROI,
    BLUR_THRESHOLD,
    RECOGNITION_CONFIDENCE_THRESHOLD,
    RECOGNITION_CONFIDENCE_FALLBACK,
    RECOGNITION_CONFIDENCE_ANY,
)


class VINExtractionPipeline:
    def __init__(self, use_gpu=False):
        self.ocr_engine = OCREngine()
        self.logger = logging.getLogger(__name__)



    def select_manual_roi(self, image: np.ndarray) -> np.ndarray:
        """Manually select ROI using CV2 and add padding."""
        print("\n--- Manual ROI Selection ---")
        print("1. Click and drag to select the region containing the ID.")
        print("2. Press ENTER or SPACE to confirm.")
        print("3. Press 'c' to cancel.")
        
        roi_box = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        
        x, y, w, h = roi_box
        if w == 0 or h == 0:
            return None
            
        # Add substantial padding to prevent character clipping at the edges
        # Use a combination of height-based and fixed padding
        pad_h = int(h * 0.3)
        pad_w = int(h * 0.5) # Use height as a reference for horizontal padding too
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]


    def process_image(self, image_path: str) -> str:
        """Robust ID extraction pipeline."""
        image = cv2.imread(image_path)
        if image is None:
            return f"Error: Could not read image path {image_path}."

        # 1. Image Quality Handling (Blur Detection)
        gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
        
        # Blur threshold - Reject if too low
        if variance < BLUR_THRESHOLD:
            error_msg = f"ERROR: Image rejected (Blurry). Laplacian Variance: {variance:.2f} < {BLUR_THRESHOLD}"

            print(error_msg)
            return error_msg

        # 2. Manual ROI Selection
        roi = self.select_manual_roi(image)
        if roi is None:
            return "Error: ROI selection cancelled or invalid."

        # 3. ROI Pre-normalization (Upscale for small characters - reduced from 2000 for performance)
        h_roi, w_roi = roi.shape[:2]
        target_w = 1000
        if w_roi < target_w:
            scale = target_w / w_roi
            roi = cv2.resize(roi, (int(w_roi * scale), int(h_roi * scale)), interpolation=cv2.INTER_LANCZOS4)

        # 4. Multi-Preprocessing Strategy (Optimized: evaluate variant 0 first, stop early if acceptable)
        variants = PreProcessor.get_all_variants(roi)
        
        best_result = {
            'text': "",
            'score': 0,
            'alnum_count': 0,
            'avg_confidence': 0.0,
            'blocks': [],
            'low_confidence': False,
        }
        # Track best result even below threshold (for fallback)
        best_fallback = dict(best_result)
        # Last resort: best result with any confidence (when all confidences are very low)
        best_any = dict(best_result)

        def get_left_x(b):
            bbox = b.get('bbox')
            if not bbox:
                return 0
            return min(p[0] for p in bbox[:4])

        def process_blocks(blocks, min_conf):
            valid = [b for b in blocks if b['confidence'] >= min_conf]
            if not valid:
                return None
            valid.sort(key=get_left_x)
            combined = " ".join(str(b['text']).strip() for b in valid if str(b['text']).strip())
            char_count = sum(1 for c in combined if not c.isspace())
            avg_conf = sum(b['confidence'] for b in valid) / len(valid)
            score = char_count * avg_conf
            return {
                'text': combined,
                'score': score,
                'alnum_count': char_count,
                'avg_confidence': avg_conf,
                'blocks': valid,
            }

        def is_acceptable_result(candidate):
            """Check if result is acceptable enough to stop early."""
            if not candidate:
                return False
            # Accept if confidence is high enough and has reasonable alnum count
            return (candidate['avg_confidence'] >= RECOGNITION_CONFIDENCE_THRESHOLD and 
                    candidate['alnum_count'] >= 3)

        print(f"\nEvaluating preprocessing variants (optimized: early stop)...")

        # Evaluate variant 0 first
        blocks = self.ocr_engine.extract_text(variants[0])
        candidate = process_blocks(blocks, RECOGNITION_CONFIDENCE_THRESHOLD)
        if candidate:
            best_result = {**candidate, 'low_confidence': False}
            print(f"Variant 0: Initial result (Len: {candidate['alnum_count']}, Score: {candidate['score']}, Conf: {candidate['avg_confidence']:.4f})")
        
        # Track fallback and any for variant 0
        fallback_candidate = process_blocks(blocks, RECOGNITION_CONFIDENCE_FALLBACK)
        if fallback_candidate and (
            not best_fallback.get('blocks') or
            fallback_candidate['score'] > best_fallback['score']
            or (fallback_candidate['score'] == best_fallback['score'] and fallback_candidate['avg_confidence'] > best_fallback['avg_confidence'])
        ):
            best_fallback = fallback_candidate
        any_candidate = process_blocks(blocks, RECOGNITION_CONFIDENCE_ANY)
        if any_candidate and (
            not best_any.get('blocks') or
            any_candidate['score'] > best_any['score']
            or (any_candidate['score'] == best_any['score'] and any_candidate['avg_confidence'] > best_any['avg_confidence'])
        ):
            best_any = any_candidate

        # Early stop if variant 0 is acceptable
        if is_acceptable_result(candidate):
            print(f"Early stop: Variant 0 produced acceptable result (Conf: {candidate['avg_confidence']:.4f})")
        else:
            # Run one fallback variant only (use variant 1 as fallback)
            if len(variants) > 1:
                print(f"Trying fallback variant 1...")
                blocks = self.ocr_engine.extract_text(variants[1])
                candidate = process_blocks(blocks, RECOGNITION_CONFIDENCE_THRESHOLD)
                if candidate and (
                    candidate['score'] > best_result['score']
                    or (candidate['score'] == best_result['score'] and candidate['avg_confidence'] > best_result['avg_confidence'])
                ):
                    best_result = {**candidate, 'low_confidence': False}
                    print(f"Variant 1: New best result (Len: {candidate['alnum_count']}, Score: {candidate['score']}, Conf: {candidate['avg_confidence']:.4f})")

                # Update fallback and any
                fallback_candidate = process_blocks(blocks, RECOGNITION_CONFIDENCE_FALLBACK)
                if fallback_candidate and (
                    fallback_candidate['score'] > best_fallback['score']
                    or (fallback_candidate['score'] == best_fallback['score'] and fallback_candidate['avg_confidence'] > best_fallback['avg_confidence'])
                ):
                    best_fallback = fallback_candidate
                any_candidate = process_blocks(blocks, RECOGNITION_CONFIDENCE_ANY)
                if any_candidate and (
                    any_candidate['score'] > best_any['score']
                    or (any_candidate['score'] == best_any['score'] and any_candidate['avg_confidence'] > best_any['avg_confidence'])
                ):
                    best_any = any_candidate

        # If no result met main threshold, use best above fallback and mark as low confidence
        if not best_result['blocks'] and best_fallback.get('blocks'):
            best_result = {
                **best_fallback,
                'low_confidence': True,
            }
            print(f"Using best low-confidence result (avg conf: {best_result['avg_confidence']:.4f}). Please verify.")
        # If still nothing, use best result with any confidence (last resort)
        elif not best_result['blocks'] and best_any.get('blocks'):
            best_result = {
                **best_any,
                'low_confidence': True,
            }
            print(f"Using best available result (avg conf: {best_result['avg_confidence']:.4f}). Please verify manually.")

        # 4. Result Formatting
        if not best_result['blocks']:
            return "No text detected from OCR. Try a tighter ROI, better lighting, or a clearer image."


        # Return raw decoded output (no regex or fixed-length enforcement).
        raw_text = best_result['text'].strip()
        cleaned_text = self.ocr_engine.handle_confusion(raw_text)

        def cleanup_output(text: str) -> str:
            """
            Lightweight cleanup only (no length-based filtering):
            - normalize internal whitespace
            - remove obvious repeated-character hallucinations
            """
            if not text:
                return ""
            txt = " ".join(text.strip().split())
            txt = re.sub(r"(.)\1{4,}", r"\1\1\1", txt)
            return txt

        final_text = cleanup_output(raw_text)

        print("\n" + "="*40)
        print("FINAL EXTRACTION RESULTS")
        print("="*40)
        if best_result.get('low_confidence'):
            print("(Low confidence - please verify)")
        print(f"Raw OCR Output: {raw_text}")
        print(f"Post-Cleanup Output: {final_text}")
        print(f"Cleaned (Heuristics): {cleaned_text}")
        print(f"Confidence Score: {best_result['avg_confidence']:.4f}")
        
        # Multi-line handling
        if len(best_result['blocks']) > 1:
            print("\nDetected Text Blocks:")
            for b in best_result['blocks']:
                print(f"- {b['text']} (Conf: {b['confidence']:.2f})")
        
        return final_text if final_text else raw_text

