"""OCR image preprocessing utilities."""
# Baseline: 15.1ms/crop CPU

from __future__ import annotations

import logging
import time

import cv2
import easyocr
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Apply the OCR preprocessing pipeline to a plate crop."""
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Increase contrast slightly with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Keep a fixed height to preserve the preprocessing contract expected by
    # downstream code/tests, while preserving plate aspect ratio.
    h, w = gray.shape[:2]
    target_height = 64
    scale = target_height / max(1, h)
    target_width = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(
        gray, (target_width, target_height), interpolation=interpolation
    )

    # Note: We DO NOT apply Adaptive Thresholding anymore, because EasyOCR's 
    # neural network expects natural grayscale/RGB images, not harsh binary images.
    return resized


class PlateReader:
    """Read plate text from image crops with EasyOCR."""

    def __init__(self, gpu: bool = False) -> None:
        self.reader = easyocr.Reader(["en"], gpu=gpu) # 'en' is enough for plates, 'ro' adds diacritics we don't want

    def read_plate(self, crop: np.ndarray) -> tuple[str, float]:
        started_at = time.perf_counter()
        processed = preprocess_crop(crop)
        
        # We strictly allow only uppercase letters and numbers
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        results = self.reader.readtext(
            processed, 
            detail=1, 
            paragraph=False,
            allowlist=allowlist,
            mag_ratio=1.5 # Instructs EasyOCR to upscale internally for better accuracy
        )

        if not results:
            raw_text = ""
            confidence = 0.0
        else:
            raw_text = "".join(
                r[1] for r in sorted(results, key=lambda r: r[0][0][0])
            )
            confidence = float(sum(r[2] for r in results) / len(results))

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            'OCR: raw="%s" conf=%.2f time=%.1fms',
            raw_text,
            confidence,
            elapsed_ms,
        )
        return raw_text, confidence
