"""
Quick visual smoke-test for PlateDetector.

Usage:
    python test_detector_visual.py <path_to_image>

Draws bounding boxes on the image and saves the result as 'debug_output.jpg'.
"""
import sys
import cv2
from app.core.config import settings
from app.pipeline.detector import PlateDetector

image_path = sys.argv[1] if len(sys.argv) > 1 else "test_car.jpg"

# Load image
frame = cv2.imread(image_path)
if frame is None:
    print(f"ERROR: Cannot read image '{image_path}'")
    sys.exit(1)

print(f"Image loaded: {frame.shape[1]}x{frame.shape[0]}")

# Run detector
detector = PlateDetector(settings.MODEL_PATH, settings.DETECTION_CONF)
detections = detector.detect(frame)

print(f"Detections: {len(detections)}")
for i, det in enumerate(detections):
    x, y, w, h = det["bbox"]
    conf = det["confidence"]
    print(f"  [{i}] bbox=({x}, {y}, {w}, {h})  conf={conf:.3f}")

    # Draw on image
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"{conf:.2f}", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

output_path = "debug_output.jpg"
cv2.imwrite(output_path, frame)
print(f"Saved annotated image to '{output_path}'")
