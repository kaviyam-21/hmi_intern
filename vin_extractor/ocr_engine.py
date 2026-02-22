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

# Minimum image height accepted by PaddleOCR's recognition model.
# Images shorter than this will be upscaled to avoid empty results.
_MIN_REC_HEIGHT = 32
_MIN_BOX_HEIGHT = 20


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

        # Initialize TrOCR model and processor (load once)
        try:
            model_name = "microsoft/trocr-small-printed"
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            )
            self.trocr_model.eval()  # Set to evaluation mode
            if not use_gpu:
                self.trocr_model = self.trocr_model.to('cpu')
            logging.info(f"TrOCR model loaded: {model_name}")
        except Exception as e:
            raise RuntimeError(f"TrOCR initialization failed: {e}") from e


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

        # Fixed 5-10px style margin around detected text.
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
            
            # Resize to optimal height (48-64) for TrOCR while preserving aspect ratio
            target_height = 56  # Middle of 48-64 range
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
            
            # Process and predict
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated = self.trocr_model.generate(
                    pixel_values,
                    max_length=40,
                    num_beams=3,
                    early_stopping=True,
                    output_scores=True,
                    return_dict_in_generate=True
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
            
            return (generated_text.strip(), confidence)
        except Exception as e:
            logging.debug(f"TrOCR recognition failed: {e}")
            return ("", 0.0)

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

            text, rec_conf = self._recognize_with_trocr(crop, retry_scale=1.0)
            if rec_conf < 0.85:
                text_retry, rec_conf_retry = self._recognize_with_trocr(crop, retry_scale=1.2)
                if rec_conf_retry > rec_conf:
                    text, rec_conf = text_retry, rec_conf_retry

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
