import time
import logging
from typing import Tuple

import cv2
import easyocr
import numpy as np

logger = logging.getLogger(__name__)

class PlateReader:
    """Extracts text from a license plate crop using EasyOCR."""

    def __init__(self) -> None:
        """Initialize EasyOCR once with English and Romanian."""
        # The prompt requires initializing once with ['ro', 'en'] languages.
        # gpu=False ensures we run on CPU as mentioned in acceptance criteria (or it implies we test on CPU)
        # However, typically leaving gpu=True is fine as it falls back to CPU if no GPU. Let's explicitly use CPU for predictable 300ms benchmark if they asked for CPU.
        # Wait, the prompt says "Runs in < 300ms per crop on CPU". It doesn't necessarily say to disable GPU, but just that CPU perf must be < 300ms.
        # EasyOCR initializes much faster if we specify gpu=False when testing on CPU, but for the actual app we might want GPU if available.
        # We'll just leave default (which uses GPU if available, else CPU), or pass gpu=False. Let's pass gpu=False to be safe.
        self.reader = easyocr.Reader(['ro', 'en'], gpu=False)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Applies preprocessing: grayscale -> CLAHE -> adaptive threshold.
        """
        # 1. Grayscale
        if len(crop.shape) == 3 and crop.shape[2] == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 3. Adaptive threshold
        # Using ADAPTIVE_THRESH_GAUSSIAN_C with a block size of 11 and C=2.
        thresh = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        return thresh

    def read_plate(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Reads the license plate text from the crop image.
        Returns:
            Tuple containing the raw_text (str) and confidence (float).
        """
        start_time = time.perf_counter()
        
        preprocessed = self._preprocess(crop)
        
        # run EasyOCR
        # readtext returns a list of (bbox, text, prob)
        results = self.reader.readtext(preprocessed)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Plate reading took {elapsed:.2f}ms")
        
        if not results:
            return "", 0.0
            
        # Combine texts if multiple are detected, and average confidence
        # For plates it's better to just concatenate everything 
        # and take average confidence. 
        texts = []
        confidences = []
        
        for bbox, text, prob in results:
            texts.append(text)
            confidences.append(prob)
            
        # Remove spaces to form a single continuous string (plate format)
        raw_text = "".join(texts).replace(" ", "")
        
        # Calculate mean confidence
        confidence = sum(confidences) / len(confidences)
        
        return raw_text, float(confidence)
