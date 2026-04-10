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
    """Wraps YOLOv8 detection with plate-model + vehicle-model compatibility.

    The model is loaded **once** at construction time and reused across
    all subsequent :meth:`detect` calls.

    Detection modes:
    - Single-stage: plate-trained models (typically one class) output plate boxes directly.
    - Two-stage fallback: generic YOLO vehicle classes are refined with Haar plate localization.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` YOLOv8 weights file.
    conf_threshold:
        Minimum confidence score for a detection to be kept.
    """

    VEHICLE_CLASS_IDS = {2, 3, 5, 7}
    VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}
    PLATE_CLASS_HINTS = ("plate", "licen", "registration")

    def __init__(self, model_path: str, conf_threshold: float) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )
        self._haar_available = not self.plate_cascade.empty()
        if not self._haar_available:
            logger.warning(
                "Haar cascade unavailable; vehicle detections will be used directly as a fallback."
            )

    def _class_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            normalized: dict[int, str] = {}
            for key, value in names.items():
                if isinstance(key, int):
                    normalized[key] = str(value)
                elif isinstance(key, str) and key.isdigit():
                    normalized[int(key)] = str(value)
            return normalized
        if isinstance(names, (list, tuple)):
            return {idx: str(name) for idx, name in enumerate(names)}
        return {}

    def _is_direct_plate_class(
        self, cls_id: int, class_names: dict[int, str]
    ) -> bool:
        label = class_names.get(cls_id, "").lower()
        if any(hint in label for hint in self.PLATE_CLASS_HINTS):
            return True

        # For single-class plate models class 0 is the plate class.
        # If class metadata is unavailable (e.g., mocked model in tests), keep
        # class 0 behavior for compatibility.
        if cls_id == 0 and (len(class_names) == 1 or not class_names):
            return True

        return False

    def _is_vehicle_class(self, cls_id: int, class_names: dict[int, str]) -> bool:
        if cls_id in self.VEHICLE_CLASS_IDS:
            return True
        label = class_names.get(cls_id, "").lower()
        return label in self.VEHICLE_CLASS_NAMES

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

        class_names = self._class_names()
        gray_frame: np.ndarray | None = None

        for box in boxes:
            cls_id = int(box.cls[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            confidence = float(box.conf[0])

            # Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            y_start, y_end = max(0, y), min(frame_h, y + h)
            x_start, x_end = max(0, x), min(frame_w, x + w)

            # Prevent zero-area boxes just in case
            if y_end <= y_start or x_end <= x_start:
                continue

            if self._is_direct_plate_class(cls_id, class_names):
                detections.append(
                    {
                        "bbox": (
                            int(x_start),
                            int(y_start),
                            int(x_end - x_start),
                            int(y_end - y_start),
                        ),
                        "confidence": confidence,
                    }
                )
                continue

            if not self._is_vehicle_class(cls_id, class_names):
                continue

            if not self._haar_available:
                detections.append(
                    {
                        "bbox": (
                            int(x_start),
                            int(y_start),
                            int(x_end - x_start),
                            int(y_end - y_start),
                        ),
                        "confidence": confidence,
                    }
                )
                continue

            if gray_frame is None:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray_frame[y_start:y_end, x_start:x_end]
            if roi_gray.size == 0:
                continue

            plates = self.plate_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=1,
                minSize=(15, 15),
            )
            for (px, py, pw, ph) in plates:
                detections.append(
                    {
                        "bbox": (
                            int(x_start + px),
                            int(y_start + py),
                            int(pw),
                            int(ph),
                        ),
                        "confidence": confidence,
                    }
                )

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
