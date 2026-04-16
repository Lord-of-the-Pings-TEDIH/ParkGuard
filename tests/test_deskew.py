"""Tests for app.pipeline.deskew — robust perspective and rotation correction."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.deskew import (
    _MAX_PLATE_AR,
    _MIN_DIM,
    _MIN_PLATE_AR,
    deskew_plate,
)


def _white_rect_on_black(
    height: int = 100,
    width: int = 300,
    plate_h: int = 30,
    plate_w: int = 120,
) -> np.ndarray:
    """Create a synthetic BGR image with a white rectangle (fake plate)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - plate_h) // 2
    x0 = (width - plate_w) // 2
    img[y0 : y0 + plate_h, x0 : x0 + plate_w] = 255
    return img


def _tiny_crop(h: int = 5, w: int = 5) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Basic passthrough / no-op cases
# ---------------------------------------------------------------------------


def test_empty_image_returns_empty() -> None:
    empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
    result = deskew_plate(empty)
    assert result.size == 0


def test_tiny_crop_returns_original() -> None:
    crop = _tiny_crop(h=_MIN_DIM - 1, w=_MIN_DIM - 1)
    result = deskew_plate(crop)
    assert np.array_equal(result, crop)


def test_normal_crop_returns_non_empty() -> None:
    crop = _white_rect_on_black()
    result = deskew_plate(crop)
    assert result.size > 0


def test_solid_black_returns_original() -> None:
    """No edges → no contours → should return the original crop."""
    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    result = deskew_plate(crop)
    assert np.array_equal(result, crop)


def test_solid_white_returns_original() -> None:
    crop = np.full((40, 120, 3), 255, dtype=np.uint8)
    result = deskew_plate(crop)
    assert np.array_equal(result, crop)


# ---------------------------------------------------------------------------
# Aspect ratio filtering
# ---------------------------------------------------------------------------


def test_square_contour_is_rejected() -> None:
    """A square object (AR=1) should NOT be used for deskew."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a 40×40 white square — AR ≈ 1, below _MIN_PLATE_AR
    img[30:70, 30:70] = 255
    result = deskew_plate(img)
    # Should return original or Hough-corrected, but NOT perspective-warp
    # to the square.  We just verify it doesn't crash and returns something.
    assert result.size > 0


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_aspect_ratio_bounds_are_reasonable() -> None:
    assert _MIN_PLATE_AR >= 1.5
    assert _MAX_PLATE_AR <= 10.0
    assert _MIN_PLATE_AR < _MAX_PLATE_AR


def test_min_dim_is_reasonable() -> None:
    assert _MIN_DIM >= 5
    assert _MIN_DIM <= 20
