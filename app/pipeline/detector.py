"""
app/pipeline/detector.py

YOLO-based license-plate detector.
"""

from __future__ import annotations

import cv2
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
        # Initialize Haar Cascade for license plates
        self.plate_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        )

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

        # Convert the frame to grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for box in boxes:
            # Only process vehicle classes: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
            cls_id = int(box.cls[0])
            if cls_id not in (2, 3, 5, 7):
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            confidence = float(box.conf[0])

            # Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            y_start, y_end = max(0, y), min(frame_h, y + h)
            x_start, x_end = max(0, x), min(frame_w, x + w)

            roi_gray = gray[y_start:y_end, x_start:x_end]
            if roi_gray.size == 0:
                continue

            # Detect plates within the vehicle ROI
            plates = self.plate_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=1,
                minSize=(15, 15)
            )

            for (px, py, pw, ph) in plates:
                detections.append({
                    "bbox": (int(x_start + px), int(y_start + py), int(pw), int(ph)),
                    "confidence": confidence,  # Inherit vehicle confidence
                })

        return detections
