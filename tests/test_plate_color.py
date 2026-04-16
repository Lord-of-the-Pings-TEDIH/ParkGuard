"""Tests for app.pipeline.plate_color — color-based plate classification."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.plate_color import PlateType, classify_plate_color


def _fill_bgr(height: int = 40, width: int = 120, bgr: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Create a solid-color BGR image."""
    img = np.full((height, width, 3), bgr, dtype=np.uint8)
    return img


def _white_bg_with_text(
    height: int = 40,
    width: int = 120,
    text_bgr: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Create a white-background image with colored text-like stripes."""
    img = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
    # Draw horizontal "text" stripes in the text color.
    for y in range(height // 4, height * 3 // 4):
        for x in range(width // 4, width * 3 // 4):
            img[y, x] = text_bgr
    return img


# ---------------------------------------------------------------------------
# Empty / tiny / invalid inputs
# ---------------------------------------------------------------------------


def test_empty_image_returns_unknown() -> None:
    empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
    assert classify_plate_color(empty) == PlateType.UNKNOWN


def test_tiny_image_returns_unknown() -> None:
    tiny = np.zeros((3, 3, 3), dtype=np.uint8)
    assert classify_plate_color(tiny) == PlateType.UNKNOWN


def test_grayscale_input_returns_unknown() -> None:
    gray = np.zeros((40, 120), dtype=np.uint8)
    assert classify_plate_color(gray) == PlateType.UNKNOWN


# ---------------------------------------------------------------------------
# Standard plates (white bg, black text)
# ---------------------------------------------------------------------------


def test_white_bg_black_text_is_standard() -> None:
    crop = _white_bg_with_text(text_bgr=(0, 0, 0))
    result = classify_plate_color(crop)
    assert result == PlateType.STANDARD


# ---------------------------------------------------------------------------
# Temporary plates (white bg, red text)
# ---------------------------------------------------------------------------


def test_white_bg_red_text_is_temporary() -> None:
    # Create a white plate with thin red text stripes (like real characters).
    # Text must be thin enough to NOT trigger PROBE (>35% red bg).
    crop = np.full((40, 120, 3), (255, 255, 255), dtype=np.uint8)
    # Draw thin red horizontal stripes simulating characters
    for y in [14, 15, 20, 21, 26, 27]:
        crop[y, 20:100] = (0, 0, 200)  # BGR red
    result = classify_plate_color(crop)
    assert result == PlateType.TEMPORARY


# ---------------------------------------------------------------------------
# Diplomatic plates (blue background)
# ---------------------------------------------------------------------------


def test_blue_bg_is_diplomatic() -> None:
    # Create a mostly-blue image (like a diplomatic plate).
    crop = _fill_bgr(bgr=(200, 100, 0))  # BGR: blue
    result = classify_plate_color(crop)
    assert result == PlateType.DIPLOMATIC


# ---------------------------------------------------------------------------
# Probe plates (red background)
# ---------------------------------------------------------------------------


def test_red_bg_is_probe() -> None:
    crop = _fill_bgr(bgr=(0, 0, 200))  # BGR: red
    result = classify_plate_color(crop)
    assert result == PlateType.PROBE


# ---------------------------------------------------------------------------
# Electric plates (green elements)
# ---------------------------------------------------------------------------


def test_green_elements_is_electric() -> None:
    # White background with a green stripe.
    crop = _fill_bgr(bgr=(255, 255, 255))
    # Add a green stripe across the bottom 30%
    green_start = int(crop.shape[0] * 0.7)
    crop[green_start:, :] = (0, 180, 0)  # BGR: green
    result = classify_plate_color(crop)
    assert result == PlateType.ELECTRIC


# ---------------------------------------------------------------------------
# PlateType enum values
# ---------------------------------------------------------------------------


def test_plate_type_values() -> None:
    assert PlateType.STANDARD.value == "standard"
    assert PlateType.TEMPORARY.value == "temporary"
    assert PlateType.DIPLOMATIC.value == "diplomatic"
    assert PlateType.PROBE.value == "probe"
    assert PlateType.ELECTRIC.value == "electric"
    assert PlateType.UNKNOWN.value == "unknown"
