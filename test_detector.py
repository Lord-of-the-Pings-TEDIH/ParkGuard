"""
test_detector.py

Smoke-test for PlateDetector.
Downloads a sample image, runs detection, prints results to console,
and saves annotated output + a text report to ./test_output/.

Usage:
    python test_detector.py                  # uses bundled sample image
    python test_detector.py path/to/image.jpg  # uses your own image
"""

from __future__ import annotations

import os
import sys
import textwrap
import urllib.request
from datetime import datetime

import cv2
import numpy as np

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from app.core.config import settings
from app.pipeline.detector import PlateDetector

# ── configuration ────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
SAMPLE_PATH = os.path.join(OUTPUT_DIR, "sample_input.jpg")


def ensure_sample_image(path: str) -> None:
    """Download, if missing, a sample image from Ultralytics assets."""
    if os.path.exists(path):
        return
    print(f"  ↓ Downloading sample image → {path}")
    urllib.request.urlretrieve(SAMPLE_URL, path)


def draw_detections(
    frame: np.ndarray, detections: list[dict]
) -> np.ndarray:
    """Return a copy of *frame* with bounding boxes and labels drawn."""
    out = frame.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"#{i} {conf:.2f}"
        cv2.putText(out, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def run_test(image_path: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 60

    print(separator)
    print("  PlateDetector  —  Smoke Test")
    print(separator)

    # ── 1. Load model ────────────────────────────────────────────
    print(f"\n[1/4] Loading YOLO model from: {settings.MODEL_PATH}")
    detector = PlateDetector(settings.MODEL_PATH, settings.DETECTION_CONF)
    print("      ✔ Model loaded successfully")

    # ── 2. Test on blank frame (expect 0 detections) ─────────────
    print("\n[2/4] Running on blank 640×480 frame (expect 0 detections)…")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_dets = detector.detect(blank)
    assert isinstance(blank_dets, list), "detect() must return a list"
    assert len(blank_dets) == 0, f"Expected 0 detections, got {len(blank_dets)}"
    print(f"      ✔ Got {len(blank_dets)} detections (correct)")

    # ── 3. Test on real image ────────────────────────────────────
    print(f"\n[3/4] Running on real image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"      ✖ ERROR: Cannot read '{image_path}'")
        sys.exit(1)
    print(f"      Image size: {frame.shape[1]}×{frame.shape[0]}")

    detections = detector.detect(frame)
    assert isinstance(detections, list), "detect() must return a list"
    print(f"      ✔ Got {len(detections)} detection(s)")

    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        assert isinstance(det["bbox"], tuple) and len(det["bbox"]) == 4
        assert all(isinstance(v, int) for v in det["bbox"])
        assert isinstance(conf, float)
        print(f"        [{i}]  bbox=({x:4d}, {y:4d}, {w:4d}, {h:4d})  "
              f"conf={conf:.4f}")

    # ── 4. Save outputs ──────────────────────────────────────────
    print(f"\n[4/4] Saving results to {OUTPUT_DIR}/")

    annotated_path = os.path.join(OUTPUT_DIR, "annotated_output.jpg")
    annotated = draw_detections(frame, detections)
    cv2.imwrite(annotated_path, annotated)
    print(f"      ✔ Annotated image → {annotated_path}")

    # Crop each detection
    crops_dir = os.path.join(OUTPUT_DIR, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        crop = frame[y : y + h, x : x + w]
        crop_path = os.path.join(crops_dir, f"crop_{i}.jpg")
        cv2.imwrite(crop_path, crop)
    if detections:
        print(f"      ✔ {len(detections)} crop(s) → {crops_dir}/")

    # Text report
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(textwrap.dedent(f"""\
            PlateDetector Smoke-Test Report
            {separator}
            Timestamp      : {timestamp}
            Model          : {settings.MODEL_PATH}
            Conf threshold : {settings.DETECTION_CONF}
            Image          : {image_path} ({frame.shape[1]}x{frame.shape[0]})
            Detections     : {len(detections)}

        """))
        for i, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            f.write(f"  [{i}] bbox=({x}, {y}, {w}, {h})  "
                    f"conf={det['confidence']:.4f}\n")
        f.write(f"\nBlank-frame test: PASS (0 detections)\n")
    print(f"      ✔ Text report   → {report_path}")

    # ── Done ─────────────────────────────────────────────────────
    print(f"\n{separator}")
    print("  ✔ ALL CHECKS PASSED")
    print(separator)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        img = sys.argv[1]
    else:
        ensure_sample_image(SAMPLE_PATH)
        img = SAMPLE_PATH

    run_test(img)
