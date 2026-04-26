"""Extract GPS location and quality metadata from any media file.

Works across the formats ParkGuard might receive from a Mobile-LPR operator:

    * **Images** — JPEG, TIFF, WebP, PNG (with eXIf chunk), and HEIC/HEIF
      when the optional ``pillow-heif`` plugin is registered.  Reads the
      EXIF GPS IFD via Pillow.

    * **Videos** — MP4/MOV/M4V/MKV/WebM/3GP/AVI containers via ``ffprobe``.
      We accept location tags from a wide range of devices: Apple
      QuickTime (``com.apple.quicktime.location.ISO6709``), Android
      Camera2 (``com.android.location`` and the legacy ``©xyz`` atom),
      Google Camera, generic ``location`` tags, and the Matroska
      ``LATITUDE`` / ``LONGITUDE`` pair.

The public entry point is :func:`extract_media_gps_metadata`, which
dispatches by file extension and returns a :class:`MediaGpsMetadata`
dataclass (or ``None`` when no usable GPS is present).  The legacy
:func:`extract_video_gps_metadata` and :func:`extract_video_gps` shims are
kept for callers that haven't migrated yet — they internally route through
the universal entry point so every format upgrade lands automatically.

GPMF (GoPro / DJI) metadata streams are intentionally out of scope: they
are a binary stream format that requires a dedicated parser and is best
handled by exporting GPX or having the operator transcode through GoPro's
quik first.  We log a debug message and return ``None`` for those cases.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Image extensions supported via Pillow.  HEIC/HEIF require ``pillow-heif``
# (optional dep) — when it is registered we transparently support them; when
# not we return ``None`` and the caller falls back to the next provenance
# source.  RAW formats are not supported here because vendor lockin makes a
# universal RAW EXIF reader unrealistic; convert to DNG/JPEG first.
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".jpe",
    ".tif", ".tiff",
    ".png",
    ".webp",
    ".heic", ".heif",
})

_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".m4v", ".mov", ".qt",
    ".mkv", ".webm",
    ".3gp", ".3g2",
    ".avi",
})


# ---------------------------------------------------------------------------
# ISO 6709 (used by ffprobe location tags)
# ---------------------------------------------------------------------------

# ISO 6709 H form, e.g. ``+47.2584+023.2537+202.369/``.  Captures lat, lon,
# and (optionally) altitude as signed decimal degrees / metres.
_ISO6709_RE = re.compile(
    r"""
    ^
    (?P<lat>[+-]\d+(?:\.\d+)?)
    (?P<lon>[+-]\d+(?:\.\d+)?)
    (?P<alt>[+-]\d+(?:\.\d+)?)?
    /?
    $
    """,
    re.VERBOSE,
)

# Tag keys are checked case-insensitively after the dict is normalised to
# lowercase; the spelling we list here only needs to match one casing.
_LOCATION_TAG_KEYS: tuple[str, ...] = (
    # Apple QuickTime — iOS / macOS
    "com.apple.quicktime.location.iso6709",
    "location-iso6709",
    # Generic / Android / MKV
    "location",
    # Legacy QuickTime ©xyz atom (also written by older Android builds)
    "©xyz",
    "xyz",
    # Google Camera and recent Android variants
    "com.android.location",
    "com.google.android.media.location",
)
_ACCURACY_TAG_KEYS: tuple[str, ...] = (
    "com.apple.quicktime.location.accuracy.horizontal",
    "location-horizontal-accuracy",
    "location.accuracy.horizontal",
)
_CREATION_TIME_TAG_KEYS: tuple[str, ...] = (
    "com.apple.quicktime.creationdate",
    "creation_time",
    "date",
    "date-eng",
)
# MKV / Matroska sometimes splits coordinates into two scalar tags rather
# than packing them into ISO 6709.  We treat them as a fallback when the
# combined tags are absent.
_MKV_LATITUDE_KEYS: tuple[str, ...] = ("latitude", "geo_latitude")
_MKV_LONGITUDE_KEYS: tuple[str, ...] = ("longitude", "geo_longitude")


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaGpsMetadata:
    """GPS-related fields parsed from any image or video container.

    Every field beyond ``latitude``/``longitude`` is optional because no
    single format carries every signal — iOS gives us accuracy and a
    creation timestamp, EXIF gives us heading and altitude, MKV gives us
    only lat/lon.  Treat ``None`` as "not available" rather than "zero".

    ``source_format`` is a free-form tag identifying which extractor
    produced the result (e.g. ``"exif"``, ``"quicktime"``, ``"mkv-tags"``).
    Useful for debugging GPS issues with unfamiliar devices.
    """

    latitude: float
    longitude: float
    altitude_m: float | None = None
    accuracy_m: float | None = None
    heading_deg: float | None = None
    recorded_at: datetime | None = None
    source_format: str = ""


# Backwards-compatible alias so existing imports keep working.  Older
# callers that say ``from ... import VideoGpsMetadata`` get the same class.
VideoGpsMetadata = MediaGpsMetadata


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_iso6709(text: str) -> tuple[float, float] | None:
    """Parse an ISO 6709 H location string into ``(latitude, longitude)``.

    Returns ``None`` when *text* is empty, malformed, or out of range.
    Altitude is parsed but discarded by this helper — the richer
    :func:`_parse_iso6709_full` returns it for callers that care.
    """
    parsed = _parse_iso6709_full(text)
    if parsed is None:
        return None
    return (parsed[0], parsed[1])


def _parse_iso6709_full(text: str) -> tuple[float, float, float | None] | None:
    if not text:
        return None
    match = _ISO6709_RE.match(text.strip())
    if not match:
        return None
    try:
        lat = float(match.group("lat"))
        lon = float(match.group("lon"))
        alt_raw = match.group("alt")
        alt = float(alt_raw) if alt_raw is not None else None
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    return (lat, lon, alt)


def _parse_accuracy(text: str) -> float | None:
    """Parse a positive horizontal-accuracy estimate in metres."""
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _parse_creation_time(text: str) -> datetime | None:
    """Parse a QuickTime / generic ISO 8601 timestamp."""
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def _parse_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def _first_tag(tags: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = tags.get(key.lower())
        if value:
            return value
    return None


# ---------------------------------------------------------------------------
# Video extraction (ffprobe)
# ---------------------------------------------------------------------------


def _run_ffprobe(path: str | Path) -> dict[str, str] | None:
    """Return the lowercased ``format.tags`` dict, or ``None`` on failure."""
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
    return {k.lower(): v for k, v in tags.items() if isinstance(v, str)}


def _extract_video_gps(path: str | Path) -> MediaGpsMetadata | None:
    """Read GPS from a video container via ffprobe."""
    tags = _run_ffprobe(path)
    if not tags:
        return None

    altitude: float | None = None
    raw_location = _first_tag(tags, _LOCATION_TAG_KEYS)
    if raw_location:
        parsed = _parse_iso6709_full(raw_location)
        if parsed is None:
            return None
        lat, lon, altitude = parsed
        source_format = "quicktime-iso6709"
    else:
        # MKV-style fallback: separate LATITUDE / LONGITUDE tags.
        lat_raw = _first_tag(tags, _MKV_LATITUDE_KEYS)
        lon_raw = _first_tag(tags, _MKV_LONGITUDE_KEYS)
        if lat_raw is None or lon_raw is None:
            return None
        lat = _parse_float(lat_raw)
        lon = _parse_float(lon_raw)
        if lat is None or lon is None:
            return None
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return None
        source_format = "mkv-tags"

    accuracy_raw = _first_tag(tags, _ACCURACY_TAG_KEYS)
    accuracy = _parse_accuracy(accuracy_raw) if accuracy_raw else None

    recorded_raw = _first_tag(tags, _CREATION_TIME_TAG_KEYS)
    recorded = _parse_creation_time(recorded_raw) if recorded_raw else None

    return MediaGpsMetadata(
        latitude=lat,
        longitude=lon,
        altitude_m=altitude,
        accuracy_m=accuracy,
        heading_deg=None,  # video containers don't carry per-clip heading
        recorded_at=recorded,
        source_format=source_format,
    )


# ---------------------------------------------------------------------------
# Image extraction (EXIF via Pillow)
# ---------------------------------------------------------------------------


def _exif_rational(value: Any) -> float | None:
    """Convert a Pillow rational (or numeric) value to a finite float.

    Pillow exposes EXIF rationals as ``IFDRational`` objects which support
    ``float(x)`` directly.  We still gate on ``isfinite`` to filter out
    division-by-zero artefacts that show up on a few Android devices.
    """
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _exif_dms_to_decimal(values: Any, ref: Any) -> float | None:
    """Convert an EXIF GPSLatitude/GPSLongitude triple to signed decimal."""
    if values is None:
        return None
    try:
        deg, minutes, seconds = values
    except (TypeError, ValueError):
        return None
    deg_f = _exif_rational(deg)
    min_f = _exif_rational(minutes)
    sec_f = _exif_rational(seconds)
    if deg_f is None or min_f is None or sec_f is None:
        return None
    decimal = deg_f + min_f / 60.0 + sec_f / 3600.0
    ref_str = ""
    if isinstance(ref, bytes):
        ref_str = ref.decode(errors="ignore").strip().upper()
    elif isinstance(ref, str):
        ref_str = ref.strip().upper()
    if ref_str in {"S", "W"}:
        decimal = -decimal
    return decimal


def _exif_altitude(value: Any, ref: Any) -> float | None:
    """Convert ``GPSAltitude`` + ``GPSAltitudeRef`` to signed metres."""
    altitude = _exif_rational(value) if value is not None else None
    if altitude is None:
        return None
    # ``ref`` is a single byte: 0 = above sea level, 1 = below.  Some
    # devices write it as an int, others as bytes.
    sign = 1.0
    if isinstance(ref, (bytes, bytearray)) and len(ref) >= 1 and ref[0] == 1:
        sign = -1.0
    elif isinstance(ref, int) and ref == 1:
        sign = -1.0
    return altitude * sign


def _exif_timestamp(date_stamp: Any, time_stamp: Any) -> datetime | None:
    """Combine GPSDateStamp (``YYYY:MM:DD``) + GPSTimeStamp triple → UTC datetime."""
    if not date_stamp:
        return None
    date_str = (
        date_stamp.decode(errors="ignore") if isinstance(date_stamp, bytes) else str(date_stamp)
    )
    parts = date_str.split(":")
    if len(parts) != 3:
        return None
    try:
        year, month, day = (int(p) for p in parts)
    except ValueError:
        return None
    hour = minute = 0
    second = 0.0
    if time_stamp is not None:
        try:
            h, m, s = time_stamp
        except (TypeError, ValueError):
            return None
        h_f = _exif_rational(h)
        m_f = _exif_rational(m)
        s_f = _exif_rational(s)
        if h_f is None or m_f is None or s_f is None:
            return None
        hour, minute, second = int(h_f), int(m_f), s_f
    try:
        from datetime import timezone

        return datetime(
            year, month, day, hour, minute, int(second),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def _extract_image_gps(path: str | Path) -> MediaGpsMetadata | None:
    """Read GPS from an image's EXIF block."""
    try:
        from PIL import ExifTags, Image, UnidentifiedImageError
    except ImportError:
        logger.debug("Pillow not available — cannot read image EXIF")
        return None

    try:
        with Image.open(path) as img:
            exif = img.getexif()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        logger.debug("Unable to open image %s: %s", path, exc)
        return None

    if not exif:
        return None

    # Pillow ≥9 exposes the GPSInfo IFD via ``Image.Exif.get_ifd``; older
    # builds put it under tag id 34853 of the top-level dict.
    try:
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except (AttributeError, KeyError):
        gps_ifd = exif.get(34853, {})  # 0x8825 is the GPSInfo IFD pointer
    if not gps_ifd:
        return None

    # Translate numeric tag ids to the human names Pillow knows about so
    # we can pull fields by name regardless of camera vendor.
    named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

    latitude = _exif_dms_to_decimal(
        named.get("GPSLatitude"), named.get("GPSLatitudeRef")
    )
    longitude = _exif_dms_to_decimal(
        named.get("GPSLongitude"), named.get("GPSLongitudeRef")
    )
    if latitude is None or longitude is None:
        return None
    if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
        return None

    altitude = _exif_altitude(named.get("GPSAltitude"), named.get("GPSAltitudeRef"))
    accuracy = _exif_rational(named.get("GPSHPositioningError"))
    if accuracy is not None and accuracy <= 0.0:
        accuracy = None
    heading = _exif_rational(named.get("GPSImgDirection"))
    if heading is not None:
        heading = heading % 360.0
    recorded = _exif_timestamp(
        named.get("GPSDateStamp"), named.get("GPSTimeStamp")
    )
    return MediaGpsMetadata(
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude,
        accuracy_m=accuracy,
        heading_deg=heading,
        recorded_at=recorded,
        source_format="exif",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def extract_media_gps_metadata(path: str | Path) -> MediaGpsMetadata | None:
    """Universal GPS-metadata extractor for any image or video file.

    Dispatches by file extension: image extensions go through Pillow EXIF,
    video extensions go through ffprobe.  For unknown extensions we try
    image first (cheaper, in-process) then fall back to ffprobe — this
    catches files with stripped or unusual extensions without surprising
    the caller.

    Returns ``None`` whenever no usable GPS pose can be recovered, leaving
    the upload flow to fall back to the configured default pose.
    """
    candidate = Path(path)
    suffix = candidate.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return _extract_image_gps(candidate)
    if suffix in _VIDEO_EXTENSIONS:
        return _extract_video_gps(candidate)
    return _extract_image_gps(candidate) or _extract_video_gps(candidate)


def extract_video_gps_metadata(path: str | Path) -> MediaGpsMetadata | None:
    """Backwards-compatible alias that now also handles images.

    Existing callers (e.g. the session router) keep working unchanged; new
    code should prefer :func:`extract_media_gps_metadata` for clarity.
    """
    return extract_media_gps_metadata(path)


def extract_video_gps(path: str | Path) -> tuple[float, float] | None:
    """Backwards-compatible shim returning just ``(latitude, longitude)``."""
    metadata = extract_media_gps_metadata(path)
    if metadata is None:
        return None
    return (metadata.latitude, metadata.longitude)
