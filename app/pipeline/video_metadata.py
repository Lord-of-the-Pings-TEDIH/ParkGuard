"""Extract GPS location from a video container's metadata.

Phones (iPhone in particular) tag MOV/MP4 files with an ISO 6709 location
string under ``com.apple.quicktime.location.ISO6709``.  We parse that here so
a Mobile-LPR session can use the recording position as the camera pose
without the operator typing GPS coordinates by hand.

The implementation shells out to ``ffprobe`` (already required by the
ffmpeg/opencv stack we ship).  When ffprobe is missing, no GPS tag is
present, or parsing fails, we return ``None`` and the caller falls back to
the configured default pose.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ISO 6709 H form emitted by iOS/QuickTime, e.g. ``+47.2584+023.2537+202.369/``.
# Captures lat and lon as signed decimal degrees; altitude/CRS suffixes are
# tolerated but ignored.
_ISO6709_RE = re.compile(
    r"""
    ^
    (?P<lat>[+-]\d+(?:\.\d+)?)
    (?P<lon>[+-]\d+(?:\.\d+)?)
    (?:[+-]\d+(?:\.\d+)?)?
    /?
    $
    """,
    re.VERBOSE,
)

_LOCATION_TAG_KEYS: tuple[str, ...] = (
    "com.apple.quicktime.location.ISO6709",
    "location-iso6709",
    "location",
    "©xyz",  # legacy QuickTime ©xyz atom
)


def parse_iso6709(text: str) -> tuple[float, float] | None:
    """Parse an ISO 6709 H location string into ``(latitude, longitude)``.

    Returns ``None`` when *text* is empty, malformed, or out of range.
    """
    if not text:
        return None
    match = _ISO6709_RE.match(text.strip())
    if not match:
        return None
    try:
        lat = float(match.group("lat"))
        lon = float(match.group("lon"))
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    return (lat, lon)


def extract_video_gps(path: str | Path) -> tuple[float, float] | None:
    """Return ``(latitude, longitude)`` from a video's container tags.

    Falls back to ``None`` (rather than raising) for any failure mode so the
    upload flow stays resilient: missing ffprobe, unreadable file, no GPS
    tag, or unparseable tag value all return ``None``.
    """
    if shutil.which("ffprobe") is None:
        return None

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format_tags",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
        return None

    if completed.returncode != 0 or not completed.stdout:
        return None

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None

    tags = payload.get("format", {}).get("tags") or {}
    # Tag keys vary slightly between iOS versions; normalise to lowercase so
    # we match any of the documented spellings.
    lower = {k.lower(): v for k, v in tags.items() if isinstance(v, str)}
    for key in _LOCATION_TAG_KEYS:
        raw = lower.get(key.lower())
        if raw:
            parsed = parse_iso6709(raw)
            if parsed is not None:
                return parsed
    return None
