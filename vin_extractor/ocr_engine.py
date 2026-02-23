from paddleocr import PaddleOCR
import cv2
import numpy as np
import logging
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Suppress PaddleOCR logging and fix PIR/oneDNN executor issues on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PADDLE_USE_MKLDNN'] = '0'
os.environ['PADDLE_DISABLE_DNNL'] = '1'
os.environ['MKLDNN_ENABLED'] = 'OFF'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_enable_new_ir'] = '0'
# OMP_NUM_THREADS: Let system use available cores for better CPU utilization
# Remove restriction to allow better parallelization
logging.getLogger("ppocr").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Minimum image height accepted by PaddleOCR's recognition model.
# Images shorter than this will be upscaled to avoid empty results.
_MIN_REC_HEIGHT = 32
_MIN_BOX_HEIGHT = 20

# Global shared TrOCR components (loaded once, reused across engine instances)
_GLOBAL_TROCR_MODEL = None
_GLOBAL_TROCR_PROCESSOR = None


class OCREngine:
    def __init__(self, use_gpu=False):
        """
        Initialize PaddleOCR (detection + recognition) and TrOCR.
        PaddleOCR 3.x does not support rec=False in constructor; we use full OCR
        to get bounding boxes, then replace recognition with TrOCR output.
        """
        # PaddleOCR: disable extra modules for better performance
        try:
            # Try with optimized settings (may not be available in all PaddleOCR versions)
            self.ocr = PaddleOCR(
                lang='en',
                device='cpu',
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
        except TypeError:
            # Fallback if extra parameters not supported
            try:
                self.ocr = PaddleOCR(
                    lang='en',
                    device='cpu',
                    enable_mkldnn=False
                )
            except Exception as e2:
                try:
                    self.ocr = PaddleOCR(lang='en')
                except Exception as e3:
                    raise RuntimeError(f"OCREngine init failed: {e3}") from e3
        except Exception as e:
            try:
                self.ocr = PaddleOCR(lang='en')
            except Exception as e2:
                raise RuntimeError(f"OCREngine init failed: {e2}") from e2

        # Initialize TrOCR model and processor (load once, shared globally)
        global _GLOBAL_TROCR_MODEL, _GLOBAL_TROCR_PROCESSOR
        if _GLOBAL_TROCR_MODEL is None or _GLOBAL_TROCR_PROCESSOR is None:
            try:
                model_name = "microsoft/trocr-small-printed"
                processor = TrOCRProcessor.from_pretrained(model_name)
                model = VisionEncoderDecoderModel.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True
                )
                model.eval()  # Set to evaluation mode for deterministic inference
                if not use_gpu:
                    model = model.to('cpu')
                _GLOBAL_TROCR_MODEL = model
                _GLOBAL_TROCR_PROCESSOR = processor
                logger.info(f"TrOCR model loaded: {model_name}")
            except Exception as e:
                raise RuntimeError(f"TrOCR initialization failed: {e}") from e

        self.trocr_model = _GLOBAL_TROCR_MODEL
        self.trocr_processor = _GLOBAL_TROCR_PROCESSOR


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bgr(image_np: np.ndarray) -> np.ndarray:
        """
        Ensure image is 3-channel BGR.
        PaddleOCR mobile models crash on grayscale (2-D) arrays.
        """
        if image_np.ndim == 2:
            return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        if image_np.ndim == 3 and image_np.shape[2] == 1:
            return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        return image_np

    @staticmethod
    def _ensure_min_height(image_np: np.ndarray, min_h: int = _MIN_REC_HEIGHT) -> np.ndarray:
        """
        Upscale image if its height is below the recognition model's minimum.
        Images that are too small produce None results from PaddleOCR.
        Preserves aspect ratio.
        """
        h, w = image_np.shape[:2]
        if h >= min_h:
            return image_np
        scale = min_h / h
        new_w = max(1, int(w * scale))
        return cv2.resize(image_np, (new_w, min_h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _parse_result(result) -> list:
        """
        Parse PaddleOCR full result into list of dicts with text, confidence, bbox.
        Supports both legacy format and PaddleOCR v3 OCRResult format.
        """
        if not result or result[0] is None:
            return []

        page = result[0]
        blocks = []

        # PaddleOCR v3 OCRResult (dict-like with dt_polys, rec_texts, rec_scores)
        if hasattr(page, "keys") and "dt_polys" in page:
            polys = page.get("dt_polys", [])
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            for i, poly in enumerate(polys):
                try:
                    bbox = poly.tolist() if hasattr(poly, "tolist") else poly
                    text = str(texts[i]) if i < len(texts) else ""
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    blocks.append({"text": text, "confidence": conf, "bbox": bbox})
                except Exception:
                    continue
            return blocks

        # Legacy fallback format: [ [ [bbox, (text, conf)], ... ] ]
        if isinstance(page, (list, tuple)):
            for entry in page:
                try:
                    if (isinstance(entry, (list, tuple)) and len(entry) == 2
                            and isinstance(entry[0], list) and isinstance(entry[1], (list, tuple)) and len(entry[1]) == 2):
                        bbox = entry[0]
                        text = str(entry[1][0])
                        conf = float(entry[1][1])
                        blocks.append({'text': text, 'confidence': conf, 'bbox': bbox})
                    elif (isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], str)):
                        blocks.append({'text': str(entry[0]), 'confidence': float(entry[1]), 'bbox': None})
                except Exception:
                    continue

        return blocks

    def _validate_bbox(self, image_np: np.ndarray, bbox: list) -> tuple[bool, list | None]:
        """Validate and clip bbox coordinates to image bounds."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return (False, None)
        try:
            pts = []
            for p in bbox[:4]:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    return (False, None)
                pts.append([float(p[0]), float(p[1])])
        except Exception:
            return (False, None)

        h, w = image_np.shape[:2]
        clipped = []
        for x, y in pts:
            cx = min(max(x, 0.0), float(w - 1))
            cy = min(max(y, 0.0), float(h - 1))
            clipped.append([cx, cy])

        xs = [p[0] for p in clipped]
        ys = [p[1] for p in clipped]
        if max(xs) <= min(xs) or max(ys) <= min(ys):
            return (False, None)
        return (True, clipped)

    @staticmethod
    def _crop_box(image_np: np.ndarray, bbox: list) -> np.ndarray:
        """Crop image region defined by bounding box coordinates."""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        # Fixed 5–10 px style margin around detected text.
        pad_x = 8
        pad_y = 6
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

        h, w = image_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return image_np[y1:y2, x1:x2]

    @staticmethod
    def _write_debug_visualization(image_np: np.ndarray, debug_items: list) -> None:
        """
        Save optional bbox debug visualization when OCR_DEBUG_VIS=1.
        """
        if os.environ.get("OCR_DEBUG_VIS", "0") != "1":
            return
        canvas = image_np.copy()
        for idx, item in enumerate(debug_items):
            bbox = item.get("bbox")
            status = item.get("status", "ok")
            if not bbox:
                continue
            color = (0, 255, 0) if status == "ok" else (0, 0, 255)
            pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, color, 2)
            cv2.putText(canvas, status, (int(bbox[0][0]), int(bbox[0][1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(canvas, str(idx), (int(bbox[0][0]), int(bbox[0][1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        out_path = os.environ.get("OCR_DEBUG_VIS_PATH", "ocr_debug_boxes.jpg")
        cv2.imwrite(out_path, canvas)

    def _recognize_with_trocr(self, crop: np.ndarray, retry_scale: float = 1.0) -> tuple:
        """
        Recognize text from cropped image using TrOCR.
        Optimized for CPU: resize to fixed text height and single-image inference.
        
        Returns: (text: str, confidence: float)
        """
        try:
            h, w = crop.shape[:2]

            # Guard for tiny crops: upscale if too small
            if h < 16 or w < 16:
                scale = max(16 / h, 16 / w)
                new_h, new_w = max(16, int(h * scale)), max(16, int(w * scale))
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                h, w = new_h, new_w
            
            # Resize to fixed height for TrOCR while preserving aspect ratio
            target_height = 64
            if h != target_height:
                scale = target_height / h
                new_h = target_height
                new_w = max(1, int(w * scale))
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply retry_scale if needed
            if retry_scale != 1.0:
                h, w = crop.shape[:2]
                new_h, new_w = int(h * retry_scale), int(w * retry_scale)
                crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert BGR to RGB for PIL
            if crop.ndim == 2:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            elif crop.ndim == 3:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                return ("", 0.0)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(crop_rgb)

            # Process and predict (deterministic beam search, no sampling)
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated = self.trocr_model.generate(
                    pixel_values,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    max_length=50,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                sequences = generated.sequences
                generated_text = self.trocr_processor.batch_decode(sequences, skip_special_tokens=True)[0]

                # Confidence from token probabilities of generated sequence (when available).
                step_scores_list = getattr(generated, "scores", None)
                if step_scores_list is not None:
                    token_probs = []
                    for step_idx, step_scores in enumerate(step_scores_list):
                        token_pos = step_idx + 1  # skip decoder start token
                        if token_pos >= sequences.shape[1]:
                            break
                        token_id = int(sequences[0, token_pos].item())
                        probs = torch.softmax(step_scores, dim=-1)
                        token_probs.append(float(probs[0, token_id].item()))
                    confidence = float(sum(token_probs) / len(token_probs)) if token_probs else 0.9
                else:
                    confidence = 0.9
            
            text_out = generated_text.strip()
            logger.debug(f"TrOCR recognition: text='{text_out}' (len={len(text_out)}), conf={confidence:.4f}")
            return (text_out, confidence)
        except Exception as e:
            logger.debug(f"TrOCR recognition failed: {e}")
            return ("", 0.0)

    def _recognize_with_sliding_window(self, crop: np.ndarray) -> tuple:
        """
        Controlled horizontal sliding-window recognition for very wide crops.
        Split into overlapping patches (25% overlap), process each once,
        and merge predictions left-to-right.
        """
        h, w = crop.shape[:2]
        if w <= 0 or h <= 0:
            return ("", 0.0)

        # Choose patch width as a fraction of the full width (e.g. 40%)
        patch_width = max(1, int(w * 0.4))
        if patch_width >= w:
            return self._recognize_with_trocr(crop, retry_scale=1.0)

        step = max(1, int(patch_width * 0.75))  # 25% overlap

        texts = []
        confs = []

        x = 0
        while x < w:
            end_x = x + patch_width
            if end_x >= w:
                end_x = w
                x = max(0, end_x - patch_width)
            patch = crop[:, x:end_x]
            if patch.size == 0:
                break
            t, c = self._recognize_with_trocr(patch, retry_scale=1.0)
            if t:
                texts.append(t)
                confs.append(c)

            if end_x == w:
                break
            x += step

        if not texts:
            return ("", 0.0)

        merged_text = "".join(texts)
        avg_conf = float(sum(confs) / len(confs))
        logger.debug(
            f"Sliding-window merge: patches={len(texts)}, merged_len={len(merged_text)}, avg_conf={avg_conf:.4f}"
        )
        return (merged_text, avg_conf)

    # ------------------------------------------------------------------
    # Public API — unchanged signature
    # ------------------------------------------------------------------

    def extract_text(self, image_np: np.ndarray) -> list:
        """
        Extract text blocks using PaddleOCR detection + TrOCR recognition.
        
        Flow:
        1. Use PaddleOCR to detect text bounding boxes (detection only)
        2. Crop each detected box
        3. Recognize each crop with TrOCR
        4. Return combined results with confidence scores
        """
        image_np = self._to_bgr(image_np)
        image_np = self._ensure_min_height(image_np)

        # Step 1: Full PaddleOCR (detection + recognition) to get bounding boxes
        try:
            result = self.ocr.ocr(image_np)
        except Exception as e:
            logging.debug(f"PaddleOCR failed: {e}")
            return []

        blocks = self._parse_result(result)
        logging.debug(f"Parsed {len(blocks)} blocks from PaddleOCR")
        if not blocks:
            return []

        # Step 2: Replace each block's text with TrOCR recognition (use Paddle boxes only)
        out_blocks = []
        debug_items = []
        for block in blocks:
            bbox = block.get('bbox')
            if bbox is None:
                out_blocks.append(block)
                continue

            is_valid, valid_bbox = self._validate_bbox(image_np, bbox)
            if not is_valid:
                debug_items.append({"bbox": bbox, "status": "invalid_bbox"})
                out_blocks.append(block)
                continue

            crop = self._crop_box(image_np, valid_bbox)
            if crop is None or crop.size == 0:
                debug_items.append({"bbox": valid_bbox, "status": "empty_crop"})
                out_blocks.append(block)
                continue

            # Reject very small detected regions.
            h_crop, w_crop = crop.shape[:2]
            if h_crop < _MIN_BOX_HEIGHT:
                debug_items.append({"bbox": valid_bbox, "status": "small_box"})
                out_blocks.append(block)
                continue
            logger.debug(
                f"Crop from bbox: width={w_crop}, height={h_crop}"
            )

            # Base recognition
            text, rec_conf = self._recognize_with_trocr(crop, retry_scale=1.0)

            # Controlled sliding window activation
            sliding_activated = False
            if w_crop > 600 or rec_conf < 0.75:
                sliding_activated = True
                text_sw, conf_sw = self._recognize_with_sliding_window(crop)
                if conf_sw > rec_conf:
                    logger.debug(
                        f"Sliding window improved confidence from {rec_conf:.4f} to {conf_sw:.4f}"
                    )
                    text, rec_conf = text_sw, conf_sw

            # Confidence stabilization: single retry with 1.1x scaled crop
            retry_triggered = False
            if rec_conf < 0.75:
                retry_triggered = True
                text_retry, rec_conf_retry = self._recognize_with_trocr(crop, retry_scale=1.1)
                if rec_conf_retry > rec_conf:
                    logger.debug(
                        f"Retry improved confidence from {rec_conf:.4f} to {rec_conf_retry:.4f}"
                    )
                    text, rec_conf = text_retry, rec_conf_retry

            if sliding_activated:
                logger.debug(
                    f"Sliding window activated for bbox with width={w_crop}, final_conf={rec_conf:.4f}"
                )
            if retry_triggered:
                logger.debug(
                    f"Confidence retry triggered, final_conf={rec_conf:.4f}"
                )

            if text:
                det_conf = block.get('confidence', 1.0)
                out_blocks.append({
                    'text': text,
                    'confidence': det_conf * rec_conf,
                    'bbox': valid_bbox,
                })
                debug_items.append({"bbox": valid_bbox, "status": "ok"})
            else:
                debug_items.append({"bbox": valid_bbox, "status": "paddle_fallback"})
                out_blocks.append(block)

        self._write_debug_visualization(image_np, debug_items)
        return out_blocks

    @staticmethod
    def handle_confusion(text: str) -> str:
        """
        Apply contextual heuristics to reduce confusion between O/0, I/1, B/8.
        - Neighbours mostly digits  → O→0, I→1, B→8
        - Neighbours mostly letters → 0→O, 1→I, 8→B
        """
        cleaned = list(text)
        for i in range(len(cleaned)):
            char = cleaned[i]
            if char.upper() not in "018OIB":
                continue

            neighbors = []
            if i > 0:
                neighbors.append(cleaned[i - 1])
            if i < len(cleaned) - 1:
                neighbors.append(cleaned[i + 1])

            digits = sum(1 for n in neighbors if n.isdigit())
            alphas = sum(1 for n in neighbors if n.isalpha())

            if char.upper() in "OIB" and digits > alphas:
                mapping = {'O': '0', 'I': '1', 'B': '8'}
                cleaned[i] = mapping.get(char.upper(), char)
            elif char in "018" and alphas > digits:
                mapping = {'0': 'O', '1': 'I', '8': 'B'}
                cleaned[i] = mapping.get(char, char)

        return "".join(cleaned)
