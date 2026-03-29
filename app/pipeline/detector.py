"""
app/pipeline/detector.py

YOLO-based license-plate detector.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO


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

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on a single BGR frame.

        Parameters
        ----------
        frame:
            BGR image array with shape ``(H, W, 3)``.

        Returns
        -------
        list[dict]
            Each dict contains:

            * ``bbox``       – ``(x, y, w, h)`` integers (top-left corner + size).
            * ``confidence`` – detection confidence as a float.

            Returns an empty list when no plates are found.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections: list[dict] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            confidence = float(box.conf[0])
            detections.append({
                "bbox": (x, y, w, h),
                "confidence": confidence,
            })

        return detections
