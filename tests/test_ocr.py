from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.pipeline.ocr import PlateReader, preprocess_crop

FIXTURES = Path(__file__).parent / "fixtures" / "crops"


def _load_crop(name: str):
    crop = cv2.imread(str(FIXTURES / name))
    assert crop is not None, f"{name} fixture missing"
    return crop


@pytest.fixture(scope="module")
def plate_reader() -> PlateReader:
    return PlateReader(gpu=False)


@pytest.mark.parametrize("fixture_name", ["clear_ro_1.jpg", "clear_ro_2.jpg"])
def test_clear_plate_returns_text_and_high_confidence(
    plate_reader: PlateReader, fixture_name: str
) -> None:
    text, confidence = plate_reader.read_plate(_load_crop(fixture_name))

    assert text.strip() != ""
    assert confidence > 0.5


def test_non_plate_returns_empty_or_low_confidence(plate_reader: PlateReader) -> None:
    text, confidence = plate_reader.read_plate(_load_crop("non_plate.jpg"))

    assert text.strip() == "" or confidence < 0.3


@pytest.mark.parametrize("fixture_name", ["clear_ro_1.jpg", "blurry_ro.jpg", "occluded_ro.jpg"])
def test_preprocess_crop_has_expected_shape(fixture_name: str) -> None:
    processed = preprocess_crop(_load_crop(fixture_name))

    assert processed.shape[0] == 64
    assert processed.shape[1] > 0
