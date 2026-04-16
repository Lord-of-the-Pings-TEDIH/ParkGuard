"""app/pipeline/plate_color.py

Color-based plate type classification for Romanian license plates.

Romanian plates have distinct color schemes per category:

  ┌──────────────┬──────────────────┬─────────────┐
  │ Type         │ Background       │ Text color  │
  ├──────────────┼──────────────────┼─────────────┤
  │ Standard     │ White            │ Black       │
  │ Temporary    │ White            │ Red         │
  │ Diplomatic   │ Blue             │ White       │
  │ Probe        │ Red              │ White       │
  │ Electric     │ White + green    │ Black       │
  └──────────────┴──────────────────┴─────────────┘

By analysing the dominant text and background colors of a plate crop
*before* running OCR, the system can constrain the FSM decoder to the
correct pattern family, eliminating ambiguity on cases like "SJ07404"
(temporary with red text) vs. "SJ07AOA" (standard with black text).

Usage
-----
    from app.pipeline.plate_color import classify_plate_color, PlateType

    plate_type = classify_plate_color(crop)
    if plate_type == PlateType.TEMPORARY:
        # ... expect county + digits only ...
"""

from __future__ import annotations

import enum
import logging
from typing import Final

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateType(enum.Enum):
    """Plate type hint derived from color analysis."""

    STANDARD = "standard"        # white bg, black text
    TEMPORARY = "temporary"      # white bg, red text
    DIPLOMATIC = "diplomatic"    # blue bg, white text
    PROBE = "probe"              # red bg, white text
    ELECTRIC = "electric"        # white bg with green stripe/elements
    UNKNOWN = "unknown"          # unclassifiable


# ---------------------------------------------------------------------------
# HSV color range definitions
#
# OpenCV HSV ranges: H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255]
# ---------------------------------------------------------------------------

# Red has two ranges in HSV (wraps around 0/180).
_RED_LOWER_1: Final[np.ndarray] = np.array([0, 80, 80], dtype=np.uint8)
_RED_UPPER_1: Final[np.ndarray] = np.array([12, 255, 255], dtype=np.uint8)
_RED_LOWER_2: Final[np.ndarray] = np.array([165, 80, 80], dtype=np.uint8)
_RED_UPPER_2: Final[np.ndarray] = np.array([180, 255, 255], dtype=np.uint8)

_BLUE_LOWER: Final[np.ndarray] = np.array([95, 80, 50], dtype=np.uint8)
_BLUE_UPPER: Final[np.ndarray] = np.array([130, 255, 255], dtype=np.uint8)

_GREEN_LOWER: Final[np.ndarray] = np.array([35, 50, 50], dtype=np.uint8)
_GREEN_UPPER: Final[np.ndarray] = np.array([85, 255, 255], dtype=np.uint8)

# Minimum fraction of pixels required to declare a color "dominant".
_MIN_COLOR_FRACTION: Final[float] = 0.05


def classify_plate_color(crop: np.ndarray) -> PlateType:
    """Classify a plate crop by its color scheme.

    Parameters
    ----------
    crop:
        BGR image of the plate crop.

    Returns
    -------
    PlateType
        The detected plate type hint.  Returns ``PlateType.UNKNOWN`` when
        the classification is not confident enough.
    """
    if crop.size == 0 or crop.ndim != 3:
        return PlateType.UNKNOWN

    h, w = crop.shape[:2]
    if h < 5 or w < 10:
        return PlateType.UNKNOWN

    # Focus on the central 60% of the crop to exclude surrounding
    # vehicle bodywork / dark background that confuses Otsu segmentation.
    margin_y = int(h * 0.20)
    margin_x = int(w * 0.20)
    roi = crop[margin_y : h - margin_y, margin_x : w - margin_x]
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 10:
        roi = crop  # fallback to full crop if margins are too aggressive

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_pixels = roi.shape[0] * roi.shape[1]

    # ------------------------------------------------------------------
    # 1. Check for blue background → Diplomatic
    # ------------------------------------------------------------------
    blue_mask = cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)
    blue_frac = float(cv2.countNonZero(blue_mask)) / total_pixels
    if blue_frac > 0.30:
        logger.debug("PlateColor: blue=%.1f%% → DIPLOMATIC", blue_frac * 100)
        return PlateType.DIPLOMATIC

    # ------------------------------------------------------------------
    # 2. Check for red-dominant background → Probe
    # ------------------------------------------------------------------
    red_mask = _red_mask(hsv)
    red_frac = float(cv2.countNonZero(red_mask)) / total_pixels
    if red_frac > 0.35:
        logger.debug("PlateColor: red_bg=%.1f%% → PROBE", red_frac * 100)
        return PlateType.PROBE

    # ------------------------------------------------------------------
    # 3. For white-background plates, separate text from background
    #    using Otsu threshold and check TEXT color.
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Text pixels = where text_mask is 255 (dark foreground, inverted).
    text_pixel_count = cv2.countNonZero(text_mask)
    if text_pixel_count < total_pixels * 0.05:
        # Not enough text pixels detected — can't classify.
        return PlateType.UNKNOWN

    # ------------------------------------------------------------------
    # 3a. Check if text is RED → Temporary
    # ------------------------------------------------------------------
    red_in_text = cv2.bitwise_and(red_mask, text_mask)
    red_text_frac = float(cv2.countNonZero(red_in_text)) / max(1, text_pixel_count)
    if red_text_frac > 0.10:
        logger.debug(
            "PlateColor: red_text=%.1f%% of text pixels → TEMPORARY",
            red_text_frac * 100,
        )
        return PlateType.TEMPORARY

    # ------------------------------------------------------------------
    # 3b. Check for green elements → Electric
    # ------------------------------------------------------------------
    green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    green_frac = float(cv2.countNonZero(green_mask)) / total_pixels
    if green_frac > _MIN_COLOR_FRACTION:
        logger.debug("PlateColor: green=%.1f%% → ELECTRIC", green_frac * 100)
        return PlateType.ELECTRIC

    # ------------------------------------------------------------------
    # 4. Default: white background + black text → Standard
    # ------------------------------------------------------------------
    # Verify background is actually light (white/near-white).
    bg_mask = cv2.bitwise_not(text_mask)
    bg_pixels = gray[bg_mask > 0]
    if len(bg_pixels) > 0:
        bg_mean_value = float(np.mean(bg_pixels))
        if bg_mean_value > 160:
            logger.debug(
                "PlateColor: white_bg(V=%.0f) + black_text → STANDARD",
                bg_mean_value,
            )
            return PlateType.STANDARD

    return PlateType.UNKNOWN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    """Create a combined red mask (red wraps around H=0/180 in HSV)."""
    mask1 = cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1)
    mask2 = cv2.inRange(hsv, _RED_LOWER_2, _RED_UPPER_2)
    return cv2.bitwise_or(mask1, mask2)


# ---------------------------------------------------------------------------
# Mapping PlateType → pattern families (for FSM hint)
# ---------------------------------------------------------------------------

# Maps a PlateType to the set of pattern *prefixes* that are valid for it.
# Used by the CTC decoder to restrict the beam to the correct family.
PLATE_TYPE_PATTERN_TAGS: dict[PlateType, frozenset[str]] = {
    PlateType.STANDARD: frozenset({"standard"}),
    PlateType.TEMPORARY: frozenset({"temporary"}),
    PlateType.DIPLOMATIC: frozenset({"diplomatic"}),
    PlateType.PROBE: frozenset({"probe"}),
    PlateType.ELECTRIC: frozenset({"electric"}),
    PlateType.UNKNOWN: frozenset(),  # empty = allow all
}
