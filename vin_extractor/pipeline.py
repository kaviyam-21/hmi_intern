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

    # ------------------------------------------------------------------
    # Generic input entrypoint (image + video)
    # ------------------------------------------------------------------

    def process_input(self, input_path: str, mode: str = "image") -> str:
        """
        Unified entrypoint for both image and video inputs.

        - mode="image": run existing single-image pipeline.
        - mode="video": run multi-frame video pipeline with fusion.
        """
        mode = (mode or "image").lower()
        if mode == "video":
            return self._process_video(input_path)
        return self.process_image(input_path)

    # ------------------------------------------------------------------
    # Video helpers (frame extraction, ROI, fusion)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_blur_score(frame: np.ndarray) -> float:
        """Compute variance of Laplacian as a blur score."""
        if frame is None or frame.size == 0:
            return 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def _apply_default_roi(frame: np.ndarray) -> np.ndarray:
        """
        Apply configured DEFAULT_ROI to a frame.

        ROI is defined as fractional coordinates [ymin, xmin, ymax, xmax].
        """
        h, w = frame.shape[:2]
        roi_cfg = DEFAULT_ROI
        y_min = int(roi_cfg["ymin"] * h)
        x_min = int(roi_cfg["xmin"] * w)
        y_max = int(roi_cfg["ymax"] * h)
        x_max = int(roi_cfg["xmax"] * w)
        y_min = max(0, min(h, y_min))
        y_max = max(0, min(h, y_max))
        x_min = max(0, min(w, x_min))
        x_max = max(0, min(w, x_max))
        if y_max <= y_min or x_max <= x_min:
            return frame
        return frame[y_min:y_max, x_min:x_max]

    def extract_frames(self, video_path: str, frame_step: int = 3, max_frames: int = 60):
        """
        Extract a subset of frames from a video for OCR.

        - Samples every `frame_step` frames to reduce redundancy.
        - Computes blur score for each sampled frame.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_step == 0:
                blur = self.compute_blur_score(frame)
                frames.append({"frame": frame, "blur": blur})
                if len(frames) >= max_frames:
                    break
            idx += 1

        cap.release()
        return frames

    def _process_single_frame_for_video(self, frame: np.ndarray) -> dict | None:
        """
        Run a lightweight variant of the existing image pipeline on a single video frame.

        Uses:
        - DEFAULT_ROI instead of manual ROI selection.
        - Same preprocessing variants & scoring approach as image mode (variant 0 + optional variant 1).
        """
        if frame is None or frame.size == 0:
            return None

        roi = self._apply_default_roi(frame)
        h_roi, w_roi = roi.shape[:2]
        if w_roi <= 0 or h_roi <= 0:
            return None

        # Upscale ROI similarly to image mode to stabilize character size.
        target_w = 1000
        if w_roi < target_w:
            scale = target_w / w_roi
            roi = cv2.resize(roi, (int(w_roi * scale), int(h_roi * scale)), interpolation=cv2.INTER_LANCZOS4)

        variants = PreProcessor.get_all_variants(roi)

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
            if not candidate:
                return False
            return (candidate['avg_confidence'] >= RECOGNITION_CONFIDENCE_THRESHOLD and
                    candidate['alnum_count'] >= 3)

        best_result = None

        # Variant 0
        blocks0 = self.ocr_engine.extract_text(variants[0])
        candidate0 = process_blocks(blocks0, RECOGNITION_CONFIDENCE_THRESHOLD)
        if candidate0:
            best_result = candidate0

        # Optional variant 1 if needed
        if (not best_result or not is_acceptable_result(best_result)) and len(variants) > 1:
            blocks1 = self.ocr_engine.extract_text(variants[1])
            candidate1 = process_blocks(blocks1, RECOGNITION_CONFIDENCE_THRESHOLD)
            if candidate1 and (
                not best_result or
                candidate1['score'] > best_result['score'] or
                (candidate1['score'] == best_result['score'] and candidate1['avg_confidence'] > best_result['avg_confidence'])
            ):
                best_result = candidate1

        return best_result

    @staticmethod
    def multi_frame_weighted_fusion(results: list[dict]) -> str:
        """
        Perform position-wise weighted character fusion across multiple frame results.

        Each element in `results` is expected to have:
            - 'text': recognized string
            - 'avg_confidence': average confidence for that frame
        """
        if not results:
            return ""

        # Filter out results with empty text
        filtered = [r for r in results if r.get('text')]
        if not filtered:
            return ""

        # Compute median length to detect extreme outliers
        lengths = sorted(len(r['text']) for r in filtered)
        mid = len(lengths) // 2
        if len(lengths) % 2 == 1:
            median_len = lengths[mid]
        else:
            median_len = (lengths[mid - 1] + lengths[mid]) / 2.0

        # Remove frames with extremely inconsistent text length
        len_min = max(1, int(median_len * 0.5))
        len_max = int(median_len * 1.5)
        consistent = [r for r in filtered if len_min <= len(r['text']) <= len_max]
        if not consistent:
            # Fallback: return highest-confidence text
            best = max(filtered, key=lambda r: r.get('avg_confidence', 0.0))
            return best.get('text', "")

        max_len = max(len(r['text']) for r in consistent)

        fused_chars = []
        for i in range(max_len):
            weight_by_char = {}
            first_occurrence_order = {}
            for idx, r in enumerate(consistent):
                text = r['text']
                if i >= len(text):
                    continue
                ch = text[i]
                conf = float(r.get('avg_confidence', 0.0))
                weight_by_char[ch] = weight_by_char.get(ch, 0.0) + conf
                if ch not in first_occurrence_order:
                    first_occurrence_order[ch] = idx

            if not weight_by_char:
                continue

            # Choose character with highest weighted score; tie → earliest frame
            best_char = None
            best_weight = -1.0
            for ch, w in weight_by_char.items():
                if w > best_weight:
                    best_char = ch
                    best_weight = w
                elif w == best_weight:
                    if first_occurrence_order.get(ch, 1e9) < first_occurrence_order.get(best_char, 1e9):
                        best_char = ch
            fused_chars.append(best_char)

        return "".join(fused_chars).strip()

    def _process_video(self, video_path: str) -> str:
        """
        Video mode:
        - Extract frames.
        - Filter by blur and OCR confidence.
        - Fuse recognized texts across frames with weighted voting.
        """
        frames = self.extract_frames(video_path)
        if not frames:
            return f"Error: Could not read video or no frames extracted from {video_path}."

        frame_results = []
        for item in frames:
            frame = item["frame"]
            blur = item["blur"]
            # Lenient blur filter for video (do not prune aggressively)
            if blur < BLUR_THRESHOLD * 0.5:
                continue
            result = self._process_single_frame_for_video(frame)
            if not result or not result.get('text'):
                continue
            # Reject very low-confidence frames
            if result['avg_confidence'] < RECOGNITION_CONFIDENCE_FALLBACK:
                continue
            frame_results.append(result)

        if not frame_results:
            return "No reliable text detected across video frames."

        fused_text = self.multi_frame_weighted_fusion(frame_results)
        if not fused_text:
            # Fall back to best single frame
            best = max(frame_results, key=lambda r: r.get('avg_confidence', 0.0))
            return best.get('text', "")

        return fused_text



    def select_manual_roi(self, image: np.ndarray) -> np.ndarray:
        """Manually select ROI using CV2 and add padding."""
        print("\n--- Manual ROI Selection ---")
        print("1. Click and drag to select the region containing the ID.")
        print("2. Press ENTER or SPACE to confirm.")
        print("3. Press 'c' to cancel.")

        try:
            roi_box = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
        except cv2.error:
            # Environment lacks GUI support (e.g. headless OpenCV build) – fall back
            # to a default ROI based on configuration instead of crashing.
            print("OpenCV GUI functions are not available in this environment.")
            print("Falling back to default ROI based on configuration.")
            h, w = image.shape[:2]
            roi_cfg = DEFAULT_ROI
            y_min = int(roi_cfg["ymin"] * h)
            x_min = int(roi_cfg["xmin"] * w)
            y_max = int(roi_cfg["ymax"] * h)
            x_max = int(roi_cfg["xmax"] * w)
            y_min = max(0, min(h, y_min))
            y_max = max(0, min(h, y_max))
            x_min = max(0, min(w, x_min))
            x_max = max(0, min(w, x_max))
            if y_max <= y_min or x_max <= x_min:
                return None
            return image[y_min:y_max, x_min:x_max]
        
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

