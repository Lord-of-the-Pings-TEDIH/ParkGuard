"""OCR image preprocessing utilities."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Apply the OCR preprocessing pipeline to a plate crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    h, w = gray.shape[:2]
    scale = 64 / h
    resized = cv2.resize(gray, (max(1, int(w * scale)), 64))

    processed = cv2.adaptiveThreshold(
        resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return processed
