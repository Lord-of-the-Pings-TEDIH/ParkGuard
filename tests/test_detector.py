"""Unit tests for PlateDetector.detect() and PlateDetector.crop_plate().

The YOLO model is mocked so tests run without weights.  The three fixture
images in tests/fixtures/ are loaded as real numpy arrays — only the model
call is faked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers to build fake YOLO result objects
# ---------------------------------------------------------------------------

def _make_box(x1: float, y1: float, x2: float, y2: float, conf: float):
    """Return a mock object that behaves like a single ultralytics Box."""
    box = MagicMock()
    box.xyxy = [np.array([x1, y1, x2, y2])]
    box.conf = [conf]
    return box


def _make_yolo_result(boxes: list | None):
    """Return a list wrapping a single mock Result whose .boxes is *boxes*."""
    result = MagicMock()
    result.boxes = boxes
    return [result]


# ---------------------------------------------------------------------------
# Fixture: a PlateDetector with a mocked YOLO model
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Return a PlateDetector whose internal YOLO model is a MagicMock."""
    with patch("app.pipeline.detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model

        from app.pipeline.detector import PlateDetector
        det = PlateDetector(model_path="fake.pt", conf_threshold=0.50)
        # Expose the mock so tests can configure return values
        det._mock_model = mock_model
        yield det


# ===================================================================
# detect() — clear plate frame (>= 1 result)
# ===================================================================

class TestDetectClearPlate:
    def test_returns_at_least_one_result(self, detector):
        frame = cv2.imread(str(FIXTURES / "clear_plate.jpg"))
        assert frame is not None, "clear_plate.jpg fixture missing"

        # Simulate YOLO finding one plate
        detector._mock_model.return_value = _make_yolo_result(
            [_make_box(500, 350, 780, 420, 0.92)]
        )

        results = detector.detect(frame)

        assert isinstance(results, list)
        assert len(results) >= 1

    def test_confidence_is_float_between_0_and_1(self, detector):
        frame = cv2.imread(str(FIXTURES / "clear_plate.jpg"))

        detector._mock_model.return_value = _make_yolo_result(
            [_make_box(500, 350, 780, 420, 0.87)]
        )

        results = detector.detect(frame)

        for det in results:
            assert isinstance(det["confidence"], float)
            assert 0.0 <= det["confidence"] <= 1.0

    def test_bbox_is_tuple_of_four_ints(self, detector):
        frame = cv2.imread(str(FIXTURES / "clear_plate.jpg"))

        detector._mock_model.return_value = _make_yolo_result(
            [_make_box(500, 350, 780, 420, 0.92)]
        )

        results = detector.detect(frame)

        for det in results:
            bbox = det["bbox"]
            assert isinstance(bbox, tuple)
            assert len(bbox) == 4
            assert all(isinstance(v, int) for v in bbox)

    def test_multiple_detections(self, detector):
        frame = cv2.imread(str(FIXTURES / "clear_plate.jpg"))

        detector._mock_model.return_value = _make_yolo_result([
            _make_box(500, 350, 780, 420, 0.92),
            _make_box(100, 200, 300, 260, 0.75),
        ])

        results = detector.detect(frame)

        assert len(results) == 2
        for det in results:
            assert 0.0 <= det["confidence"] <= 1.0


# ===================================================================
# detect() — no plate frame (empty list)
# ===================================================================

class TestDetectNoPlate:
    def test_returns_empty_list_when_boxes_none(self, detector):
        frame = cv2.imread(str(FIXTURES / "no_plate.jpg"))
        assert frame is not None, "no_plate.jpg fixture missing"

        detector._mock_model.return_value = _make_yolo_result(boxes=None)

        results = detector.detect(frame)

        assert results == []

    def test_returns_empty_list_when_boxes_empty(self, detector):
        frame = cv2.imread(str(FIXTURES / "no_plate.jpg"))

        # boxes is an iterable but empty
        detector._mock_model.return_value = _make_yolo_result(boxes=[])

        results = detector.detect(frame)

        assert results == []


# ===================================================================
# detect() — partial plate frame
# ===================================================================

class TestDetectPartialPlate:
    def test_partial_plate_still_returned_if_above_threshold(self, detector):
        frame = cv2.imread(str(FIXTURES / "partial_plate.jpg"))
        assert frame is not None, "partial_plate.jpg fixture missing"

        # Model returns the partial plate with lower confidence
        detector._mock_model.return_value = _make_yolo_result(
            [_make_box(1180, 400, 1280, 460, 0.55)]
        )

        results = detector.detect(frame)

        assert len(results) == 1
        assert results[0]["confidence"] == pytest.approx(0.55)


# ===================================================================
# detect() — passes conf threshold to model
# ===================================================================

class TestDetectThreshold:
    def test_conf_threshold_forwarded_to_model(self, detector):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector._mock_model.return_value = _make_yolo_result(boxes=None)

        detector.detect(frame)

        detector._mock_model.assert_called_once_with(
            frame, conf=0.50, verbose=False
        )


# ===================================================================
# crop_plate()
# ===================================================================

class TestCropPlate:
    def test_basic_crop(self):
        from app.pipeline.detector import PlateDetector

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[350:420, 500:780] = 255  # white plate region

        crop = PlateDetector.crop_plate(frame, (500, 350, 280, 70), padding=0)

        assert crop.shape == (70, 280, 3)
        assert crop.mean() == 255.0  # all white

    def test_padding_expands_crop(self):
        from app.pipeline.detector import PlateDetector

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        crop = PlateDetector.crop_plate(frame, (500, 350, 280, 70), padding=10)

        # 70 + 2*10 = 90 height, 280 + 2*10 = 300 width
        assert crop.shape == (90, 300, 3)

    def test_padding_clamped_to_frame_bounds(self):
        from app.pipeline.detector import PlateDetector

        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        crop = PlateDetector.crop_plate(frame, (0, 0, 50, 30), padding=20)

        # y: max(0-20,0)=0 to min(30+20,100)=50 -> 50
        # x: max(0-20,0)=0 to min(50+20,200)=70 -> 70
        assert crop.shape == (50, 70, 3)

    def test_raises_valueerror_on_zero_size_bbox(self):
        from app.pipeline.detector import PlateDetector

        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Empty crop"):
            PlateDetector.crop_plate(frame, (200, 100, 0, 0), padding=0)

    def test_raises_valueerror_on_bbox_outside_frame(self):
        from app.pipeline.detector import PlateDetector

        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Empty crop"):
            PlateDetector.crop_plate(frame, (300, 300, 50, 50), padding=0)


# ===================================================================
# save_crop()
# ===================================================================

class TestSaveCrop:
    def test_saves_valid_jpeg_and_returns_relative_path(self, tmp_path):
        from app.pipeline.detector import PlateDetector

        crop = np.full((70, 280, 3), 128, dtype=np.uint8)
        detection_id = "abc-123"

        rel_path = PlateDetector.save_crop(crop, str(tmp_path), detection_id)

        # Correct return value
        assert rel_path == "crops/abc-123.jpg"

        # File exists on disk
        saved = tmp_path / "abc-123.jpg"
        assert saved.exists()
        assert saved.stat().st_size > 0

        # File is valid JPEG (starts with FFD8)
        header = saved.read_bytes()[:2]
        assert header == b"\xff\xd8"

        # Readable by OpenCV with correct shape
        loaded = cv2.imread(str(saved))
        assert loaded is not None
        assert loaded.shape == (70, 280, 3)

    def test_raises_ioerror_on_bad_directory(self):
        from app.pipeline.detector import PlateDetector

        crop = np.full((70, 280, 3), 128, dtype=np.uint8)

        with pytest.raises(IOError, match="Failed to write crop"):
            PlateDetector.save_crop(crop, "/nonexistent/dir", "det-1")
