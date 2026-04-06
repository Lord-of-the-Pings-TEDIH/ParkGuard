"""
app/pipeline/detector.py

YOLO-based license-plate detector.

# Baseline: ~22ms/frame on CPU (YOLOv8n, 1280x720), CUDA not benchmarked yet.
"""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class PlateDetector:
    """Wraps a YOLOv8 model for license-plate detection.

    The model is loaded **once** at construction time and reused across
    all subsequent :meth:`detect` calls.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` YOLOv8 weights file.
    conf_threshold:
        Minimum confidence score for a detection to be kept.
    """

    def __init__(self, model_path: str, conf_threshold: float) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[dict]:
        """Run inference on a single BGR frame.

        Parameters
        ----------
        frame:
            BGR image array with shape ``(H, W, 3)``.
        frame_index:
            Optional index used only for logging.

        Returns
        -------
        list[dict]
            Each dict contains:

            * ``bbox``       – ``(x, y, w, h)`` integers (top-left corner + size).
            * ``confidence`` – detection confidence as a float.

            Returns an empty list when no plates are found.
        """
        t0 = time.perf_counter()
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections: list[dict] = []
        boxes = results[0].boxes
        if boxes is None:
            logger.debug(
                "Frame %d: YOLO inference %.1fms, 0 detections",
                frame_index, elapsed_ms,
            )
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            confidence = float(box.conf[0])
            detections.append({
                "bbox": (x, y, w, h),
                "confidence": confidence,
            })

        logger.debug(
            "Frame %d: YOLO inference %.1fms, %d detections",
            frame_index, elapsed_ms, len(detections),
        )

        return detections

    @staticmethod
    def crop_plate(
        frame: np.ndarray, bbox: tuple, padding: int = 4
    ) -> np.ndarray:
        """Extract the plate region from *frame* with optional padding.

        Parameters
        ----------
        frame:
            BGR image array with shape ``(H, W, 3)``.
        bbox:
            ``(x, y, w, h)`` integers (top-left corner + size).
        padding:
            Extra pixels to include around the bounding box.

        Returns
        -------
        np.ndarray
            Cropped BGR image.

        Raises
        ------
        ValueError
            If the resulting crop is empty (zero-area).
        """
        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img_w)
        y2 = min(y + h + padding, img_h)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError("Empty crop")

        return crop

    @staticmethod
    def save_crop(crop: np.ndarray, crops_dir: str, detection_id: str) -> str:
        """Write *crop* as a JPEG and return the relative path.

        Parameters
        ----------
        crop:
            BGR image array to save.
        crops_dir:
            Absolute or project-relative directory for crop files.
        detection_id:
            Unique identifier used as the filename stem.

        Returns
        -------
        str
            Relative path in the form ``crops/{detection_id}.jpg``.

        Raises
        ------
        IOError
            If ``cv2.imwrite`` fails.
        """
        filename = f"{detection_id}.jpg"
        path = os.path.join(crops_dir, filename)
        ok = cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise IOError(f"Failed to write crop to {path}")
        return f"crops/{filename}"
