from __future__ import annotations

import numpy as np
import pytest

import app.pipeline.ocr as ocr_module
from app.pipeline.ocr import PlateReader, preprocess_crop, preprocess_clahe, preprocess_adaptive, preprocess_inverted


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


# ---------------------------------------------------------------------------
# Multi-pass preprocessing variants
# ---------------------------------------------------------------------------


def test_preprocess_clahe_returns_64px_height() -> None:
    result = preprocess_clahe(_synthetic_crop())
    assert result.ndim == 2
    assert result.shape[0] == 64


def test_preprocess_adaptive_returns_64px_height() -> None:
    result = preprocess_adaptive(_synthetic_crop())
    assert result.ndim == 2
    assert result.shape[0] == 64


def test_preprocess_inverted_returns_64px_height() -> None:
    result = preprocess_inverted(_synthetic_crop())
    assert result.ndim == 2
    assert result.shape[0] == 64


def test_all_preprocess_variants_agree_on_shape() -> None:
    crop = _synthetic_crop(height=30, width=100)
    r1 = preprocess_clahe(crop)
    r2 = preprocess_adaptive(crop)
    r3 = preprocess_inverted(crop)
    assert r1.shape == r2.shape == r3.shape


# ---------------------------------------------------------------------------
# FSM Beam Search integration
# ---------------------------------------------------------------------------


def test_read_plate_with_fsm_corrects_digit_in_letter_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = _FakePaddleReader(
        [{"rec_texts": ["CJ12A8C"], "rec_scores": [0.9]}]
    )
    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: fake_reader,
    )

    plate_reader = PlateReader(gpu=False, use_fsm_decode=True, fsm_beam_width=5)
    text, _ = plate_reader.read_plate(_synthetic_crop())

    assert text == "CJ12ABC"


def test_read_plate_with_fsm_corrects_letter_in_digit_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = _FakePaddleReader(
        [{"rec_texts": ["B1ZABC"], "rec_scores": [0.85]}]
    )
    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: fake_reader,
    )

    plate_reader = PlateReader(gpu=False, use_fsm_decode=True)
    text, _ = plate_reader.read_plate(_synthetic_crop())

    assert text == "B12ABC"


def test_read_plate_fsm_disabled_does_not_alter_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = _FakePaddleReader(
        [{"rec_texts": ["CJ12A8C"], "rec_scores": [0.9]}]
    )
    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: fake_reader,
    )

    plate_reader = PlateReader(gpu=False, use_fsm_decode=False)
    text, _ = plate_reader.read_plate(_synthetic_crop())

    assert text == "CJ12A8C"


# ---------------------------------------------------------------------------
# Multi-pass OCR integration
# ---------------------------------------------------------------------------


def test_read_plate_multi_pass_picks_best_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-pass should run OCR 3 times and pick the best valid result.

    Early-exit is disabled (``early_exit_conf=2.0``) so we observe the full
    pass sweep — the behaviour the test was originally written for.  See
    ``test_read_plate_multi_pass_early_exit`` for the short-circuit path.
    """
    call_count = 0

    class _FakeMultipassReader:
        def predict(self, _img):
            nonlocal call_count
            call_count += 1
            # All passes return the same valid plate
            return [{"rec_texts": ["B12ABC"], "rec_scores": [0.9]}]

    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: _FakeMultipassReader(),
    )

    plate_reader = PlateReader(
        gpu=False, use_multi_pass=True, early_exit_conf=2.0
    )
    text, confidence = plate_reader.read_plate(_synthetic_crop())

    assert text == "B12ABC"
    assert confidence > 0
    # Multi-pass should have called OCR 3 times (clahe, adaptive, inverted)
    assert call_count == 3


def test_read_plate_multi_pass_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high-confidence valid plate on the first pass skips the rest."""
    call_count = 0

    class _FakeReader:
        def predict(self, _img):
            nonlocal call_count
            call_count += 1
            return [{"rec_texts": ["B12ABC"], "rec_scores": [0.95]}]

    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: _FakeReader(),
    )

    plate_reader = PlateReader(
        gpu=False, use_multi_pass=True, early_exit_conf=0.85
    )
    text, _confidence = plate_reader.read_plate(_synthetic_crop())

    assert text == "B12ABC"
    # CLAHE pass returned valid + conf >= early_exit_conf — the adaptive
    # and inverted passes must NOT run.
    assert call_count == 1


def test_read_plate_multi_pass_disabled_calls_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    class _FakeSingleReader:
        def predict(self, _img):
            nonlocal call_count
            call_count += 1
            return [{"rec_texts": ["B12ABC"], "rec_scores": [0.9]}]

    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: _FakeSingleReader(),
    )

    plate_reader = PlateReader(gpu=False, use_multi_pass=False)
    plate_reader.read_plate(_synthetic_crop())

    assert call_count == 1


# ---------------------------------------------------------------------------
# Super-Resolution integration (mocked resolver)
# ---------------------------------------------------------------------------


def test_read_plate_super_resolver_called_on_small_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = _FakePaddleReader(
        [{"rec_texts": ["B12ABC"], "rec_scores": [0.9]}]
    )
    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: fake_reader,
    )

    call_log: list[np.ndarray] = []

    class _MockResolver:
        enabled = True

        def upscale_if_small(self, img: np.ndarray, *, min_height: int) -> np.ndarray:
            call_log.append(img)
            return img

    plate_reader = PlateReader(gpu=False)
    plate_reader._super_resolver = _MockResolver()  # type: ignore[assignment]
    plate_reader._sr_min_height = 32

    plate_reader.read_plate(_synthetic_crop(height=20, width=80))

    assert len(call_log) == 1


def test_read_plate_super_resolver_not_called_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_reader = _FakePaddleReader(
        [{"rec_texts": ["B12ABC"], "rec_scores": [0.9]}]
    )
    monkeypatch.setattr(
        ocr_module,
        "_build_paddle_ocr",
        lambda **_kwargs: fake_reader,
    )

    plate_reader = PlateReader(gpu=False, use_super_res=False)
    assert plate_reader._super_resolver is None
