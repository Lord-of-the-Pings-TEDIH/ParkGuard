from __future__ import annotations

import numpy as np
import pytest

import app.pipeline.ocr as ocr_module
from app.pipeline.ocr import PlateReader, preprocess_crop


class _FakePaddleReader:
    def __init__(self, result):
        self._result = result

    def predict(self, _img):
        return self._result


def _synthetic_crop(height: int = 60, width: int = 200) -> np.ndarray:
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[:, ::2] = 255
    return crop


def test_read_plate_normalizes_text_and_averages_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = _FakePaddleReader(
        [{"rec_texts": ["B 12", "abc"], "rec_scores": [0.9, 0.7]}]
    )
    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: fake_reader,
    )

    plate_reader = PlateReader(gpu=False)
    text, confidence = plate_reader.read_plate(_synthetic_crop())

    assert text == "B12ABC"
    assert confidence == pytest.approx(0.8)


def test_read_plate_returns_empty_when_no_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = _FakePaddleReader([])
    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: fake_reader,
    )

    plate_reader = PlateReader(gpu=False)
    text, confidence = plate_reader.read_plate(_synthetic_crop())

    assert text == ""
    assert confidence == 0.0


@pytest.mark.parametrize(("height", "width"), [(30, 100), (64, 64), (90, 220)])
def test_preprocess_crop_has_expected_shape(height: int, width: int) -> None:
    processed = preprocess_crop(_synthetic_crop(height=height, width=width))

    assert processed.ndim == 2
    assert processed.shape[0] == 64
    assert processed.shape[1] > 0
