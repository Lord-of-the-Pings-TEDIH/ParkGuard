"""
app/pipeline/detector.py

Ultralytics YOLO-based license-plate detector (supports YOLO11 fine-tuned models).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, TypedDict

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DetectionResult(TypedDict):
    bbox: tuple[int, int, int, int]
    confidence: float


class PlateDetector:
    """Wraps Ultralytics YOLO detection for plate-trained models.

    The model is loaded **once** at construction time and reused across
    all subsequent :meth:`detect` calls.

    This detector intentionally assumes a dedicated plate model (for example,
    a fine-tuned YOLO11 checkpoint).

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` Ultralytics weights file (e.g. YOLO11 fine-tuned model).
    conf_threshold:
        Minimum confidence score for a detection to be kept.
    """

    PLATE_CLASS_HINTS = ("plate", "licen", "registration")

    def __init__(self, model_path: str, conf_threshold: float) -> None:
        self.model = YOLO(model_path)
        normalized_conf = float(conf_threshold)
        if normalized_conf < 0.0 or normalized_conf > 1.0:
            logger.warning(
                "conf_threshold %.3f out of [0, 1]; clamping to valid interval",
                normalized_conf,
            )
        self.conf_threshold = max(0.0, min(1.0, normalized_conf))

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

        # For single-class plate models (e.g. YOLO11 fine-tuned for plates),
        # class 0 is treated as the plate class.
        # If class metadata is unavailable (e.g., mocked model in tests), keep
        # class 0 behavior for compatibility.
        if cls_id == 0 and (len(class_names) == 1 or not class_names):
            return True

        return False

    @staticmethod
    def _to_scalar(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return float(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return PlateDetector._to_scalar(value[0])
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        return None

    @classmethod
    def _extract_box_values(
        cls, box: Any
    ) -> tuple[int, float, float, float, float, float] | None:
        cls_id_val = cls._to_scalar(getattr(box, "cls", None))
        conf_val = cls._to_scalar(getattr(box, "conf", None))
        xyxy = getattr(box, "xyxy", None)
        if xyxy is None or cls_id_val is None or conf_val is None:
            return None

        coords = np.asarray(xyxy.cpu() if hasattr(xyxy, "cpu") else xyxy)
        if coords.size < 4:
            return None
        flat = coords.reshape(-1)
        x1, y1, x2, y2 = (
            float(flat[0]),
            float(flat[1]),
            float(flat[2]),
            float(flat[3]),
        )
        return int(cls_id_val), float(conf_val), x1, y1, x2, y2

    @staticmethod
    def _clip_xyxy_to_bbox(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int] | None:
        left = max(0, int(np.floor(min(x1, x2))))
        top = max(0, int(np.floor(min(y1, y2))))
        right = min(frame_w, int(np.ceil(max(x1, x2))))
        bottom = min(frame_h, int(np.ceil(max(y1, y2))))
        if right <= left or bottom <= top:
            return None
        return left, top, right - left, bottom - top

    @staticmethod
    def _append_detection(
        out: list[DetectionResult],
        *,
        bbox: tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        out.append(
            {
                "bbox": bbox,
                "confidence": max(0.0, min(1.0, float(confidence))),
            }
        )

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[DetectionResult]:
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
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            logger.warning("Frame %d: invalid frame input; skipping detection", frame_index)
            return []

        frame_h, frame_w = frame.shape[:2]
        if frame_h == 0 or frame_w == 0:
            logger.warning("Frame %d: empty frame dimensions; skipping detection", frame_index)
            return []

        t0 = time.perf_counter()
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections: list[DetectionResult] = []
        if not results:
            logger.debug("Frame %d: YOLO returned no result objects", frame_index)
            return detections

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            logger.debug(
                "Frame %d: YOLO inference %.1fms, 0 detections",
                frame_index, elapsed_ms,
            )
            return detections

        class_names = self._class_names()

        for box in boxes:
            parsed = self._extract_box_values(box)
            if parsed is None:
                logger.debug("Frame %d: skipping malformed YOLO box", frame_index)
                continue

            cls_id, confidence, x1, y1, x2, y2 = parsed
            if confidence < self.conf_threshold:
                continue

            clipped = self._clip_xyxy_to_bbox(
                x1, y1, x2, y2, frame_w=frame_w, frame_h=frame_h
            )
            if clipped is None:
                continue
            x_start, y_start, width, height = clipped

            if self._is_direct_plate_class(cls_id, class_names):
                self._append_detection(
                    detections,
                    bbox=(x_start, y_start, width, height),
                    confidence=confidence,
                )

        detections.sort(key=lambda det: det["confidence"], reverse=True)

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
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise ValueError("Invalid frame")
        if len(bbox) != 4:
            raise ValueError("bbox must have 4 elements")

        x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        if w <= 0 or h <= 0:
            raise ValueError("Empty crop")

        padding = max(0, int(padding))
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
