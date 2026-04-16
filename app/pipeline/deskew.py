"""app/pipeline/deskew.py

Robust perspective and rotation correction for license-plate crops.

Strategy (in priority order):
  1. **Contour-based perspective warp** — finds a 4-point plate outline via
     cascading epsilon values and aspect-ratio / area filtering.
  2. **Hough-line rotation fallback** — when no valid contour is found,
     estimates the dominant line angle and applies a simple rotation.
  3. **Passthrough** — returns the original crop untouched when neither
     method produces a useful result.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Plate aspect-ratio bounds (width / height).
# Real Romanian plates range roughly from 3:1 to 5.5:1.
# We use generous bounds to tolerate imprecise contours.
_MIN_PLATE_AR: float = 2.0
_MAX_PLATE_AR: float = 7.0

# Minimum fraction of crop area that the detected contour must cover.
_MIN_AREA_RATIO: float = 0.15

# Epsilon values tried in descending precision.  Smaller epsilon = tighter
# polygon approximation → better chance of finding exactly 4 vertices.
_EPSILONS: tuple[float, ...] = (0.02, 0.03, 0.05)

# Minimum crop dimensions below which deskew is skipped entirely.
_MIN_DIM: int = 10


def deskew_plate(crop: np.ndarray) -> np.ndarray:
    """Transform a skewed plate crop into a flat rectangle.

    Uses contour detection with cascading epsilon values and aspect-ratio
    filtering.  Falls back to Hough-line rotation when no valid 4-point
    contour is found.

    Parameters
    ----------
    crop:
        BGR image array.

    Returns
    -------
    np.ndarray
        De-skewed BGR image, or the original crop when correction is not
        possible.
    """
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if h < _MIN_DIM or w < _MIN_DIM:
        return crop

    crop_area = h * w

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # ------------------------------------------------------------------
    # 1. Contour-based perspective warp (cascading epsilon)
    # ------------------------------------------------------------------
    best_quad: np.ndarray | None = None
    best_score: float = 0.0

    for eps_frac in _EPSILONS:
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps_frac * peri, True)

            if len(approx) != 4:
                continue

            area = cv2.contourArea(approx)
            area_ratio = area / max(1, crop_area)
            if area_ratio < _MIN_AREA_RATIO:
                continue

            # Check aspect ratio of the bounding rectangle.
            rect = cv2.minAreaRect(approx)
            rect_w, rect_h = rect[1]
            if rect_w == 0 or rect_h == 0:
                continue
            ar = max(rect_w, rect_h) / min(rect_w, rect_h)
            if ar < _MIN_PLATE_AR or ar > _MAX_PLATE_AR:
                continue

            # Score by area (prefer larger, more confident quads).
            score = area_ratio * ar  # bigger + more plate-shaped = better
            if score > best_score:
                best_score = score
                best_quad = approx

        if best_quad is not None:
            break  # Found with tighter epsilon — no need to try looser ones

    if best_quad is not None:
        warped = _perspective_warp(crop, best_quad)
        if warped is not None and warped.size > 0:
            return warped

    # ------------------------------------------------------------------
    # 2. Hough-line rotation fallback
    # ------------------------------------------------------------------
    angle = _estimate_skew_angle(edged)
    if angle is not None and abs(angle) > 0.5:
        logger.debug("Deskew: Hough fallback angle=%.1f°", angle)
        return _rotate_simple(crop, angle)

    return crop


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _perspective_warp(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """Warp *image* so that *quad* maps to a tight rectangle."""
    pts = quad.reshape(4, 2).astype("float32")

    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    tl, tr, br, bl = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    if max_width < 4 or max_height < 4:
        return None

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def _estimate_skew_angle(edges: np.ndarray) -> float | None:
    """Estimate the dominant skew angle from Canny edges using Hough lines.

    Returns the median angle in degrees (negative = clockwise skew), or
    ``None`` if no usable lines are found.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=edges.shape[1] // 4,
        maxLineGap=10,
    )
    if lines is None or len(lines) == 0:
        return None

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 2:
            continue  # Nearly vertical — not useful for plate skew
        angle_deg = math.degrees(math.atan2(dy, dx))
        # Only keep near-horizontal lines (plates are roughly horizontal).
        if abs(angle_deg) < 30:
            angles.append(angle_deg)

    if not angles:
        return None

    angles.sort()
    return angles[len(angles) // 2]  # Median angle


def _rotate_simple(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *image* by *angle* degrees around its centre."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
