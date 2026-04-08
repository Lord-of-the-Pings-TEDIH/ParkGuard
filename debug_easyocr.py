"""
Throwaway EasyOCR smoke test.

Usage:
    python debug_easyocr.py [path_to_crop_image]
"""

from __future__ import annotations

import os
import sys

import cv2
import easyocr
import numpy as np


def load_crop_image() -> tuple[np.ndarray, str]:
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        crop_image = cv2.imread(image_path)
        if crop_image is None:
            raise SystemExit(f"Cannot read image: {image_path}")
        return crop_image, image_path

    default_path = os.path.join(os.path.dirname(__file__), "debug_output.jpg")
    if os.path.exists(default_path):
        crop_image = cv2.imread(default_path)
        if crop_image is not None:
            return crop_image, default_path

    crop_image = np.full((120, 360, 3), 255, dtype=np.uint8)
    cv2.putText(
        crop_image,
        "B123ABC",
        (18, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    return crop_image, "<synthetic>"


def main() -> None:
    crop_image, image_source = load_crop_image()
    print(f"Image source: {image_source}")

    reader = easyocr.Reader(["ro", "en"], gpu=False)
    result = reader.readtext(crop_image)
    print("Result:")
    print(result)

    if not isinstance(result, list):
        raise SystemExit(f"Expected list output from EasyOCR, got: {type(result).__name__}")

    for idx, item in enumerate(result):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise SystemExit(f"Unexpected OCR item at index {idx}: {item!r}")
        bbox, text, confidence = item
        print(f"[{idx}] bbox={bbox} text={text!r} confidence={confidence}")


if __name__ == "__main__":
    main()
