"""Validate GPS coordinates obtained for a Mobile-LPR session.

GPS is never "100% correct" — even commercial-grade hardware has 3–5 m of
horizontal error and consumer phones can drift much further when the fix is
cold or the sky is obstructed.  What this module does is ring-fence the
sources we *do* trust by:

    * **Hard validation** — reject coordinates that cannot possibly be
      a valid Romanian GPS fix (NaN/inf, out of WGS84 range, Null Island,
      or outside the Romanian bounding box).  These raise
      :class:`GpsValidationError` because letting them through would silently
      project every plate to a wrong spot.

    * **Soft validation** — when two independent sources are available
      (e.g. explicit user input + the GPS embedded in the video container),
      we cross-check them with the haversine distance and emit a warning
      when they disagree by more than ``CROSS_SOURCE_TOLERANCE_M``.  The
      caller decides whether to use the explicit reading (preferred) or
      surface the warning to the operator.

    * **Provenance tracking** — every resolved pose carries a
      :class:`GpsSource` tag (``EXPLICIT``, ``VIDEO_METADATA``,
      ``CONFIG_DEFAULT``) so downstream code and the UI can show *where*
      the coordinates came from and how much to trust them.

Romania is the operational region for ParkGuard's parking-spot dataset, so
all in-country checks use the country bounding box defined here.  If the
project ever expands beyond Romania this becomes a configurable allow-list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from geopy.distance import geodesic


# Romania bounding box (WGS84).  Generously rounded outward from the
# extremes (Mihai Bravu N=48.27, Zimnicea S=43.62, Beba Veche W=20.26,
# Sulina E=29.71) so legitimate GPS noise near the border still passes.
ROMANIA_LAT_MIN: float = 43.5
ROMANIA_LAT_MAX: float = 48.4
ROMANIA_LON_MIN: float = 20.2
ROMANIA_LON_MAX: float = 29.8

# Two independent GPS readings (e.g. explicit form input vs. the iOS
# container fix) for the same vehicle should agree to within this many
# metres.  Tuned for "operator typed coords, then uploaded the video taken
# at that spot" — a 50 m gap is consistent with phone GPS drift, anything
# larger almost always means the operator paired the wrong video with the
# wrong location.
CROSS_SOURCE_TOLERANCE_M: float = 50.0


class GpsSource(str, Enum):
    """Where the resolved GPS pose came from.

    Values are stored verbatim in the ``Session.gps_source`` column so the
    frontend can render a provenance badge next to the coordinates.
    """

    EXPLICIT = "explicit"
    VIDEO_METADATA = "video_metadata"
    CONFIG_DEFAULT = "config_default"


class GpsValidationStatus(str, Enum):
    """Outcome of GPS validation for a session."""

    OK = "ok"
    WARNING = "warning"
    INVALID = "invalid"


class GpsValidationError(ValueError):
    """Raised when GPS coordinates fail a hard validation check.

    Hard failures (out-of-range, NaN, Null Island, outside Romania) cannot
    be coerced into a usable pose — surfacing this as a 400 to the API
    caller is preferable to silently projecting plates to garbage spots.
    """


@dataclass(frozen=True)
class ValidatedCoordinates:
    """A latitude/longitude pair that has passed every hard validation rule."""

    latitude: float
    longitude: float


@dataclass
class ResolvedGpsPose:
    """A fully-resolved Mobile-LPR pose with provenance + soft warnings."""

    latitude: float
    longitude: float
    heading_deg: float
    source: GpsSource
    status: GpsValidationStatus = GpsValidationStatus.OK
    accuracy_m: float | None = None
    warnings: list[str] = field(default_factory=list)


def _is_finite(value: float) -> bool:
    """Return True only for real, finite floats (rejects NaN and ±inf)."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_latitude(value: float) -> float:
    """Return ``value`` if it is a finite WGS84 latitude, else raise."""
    if not _is_finite(value):
        raise GpsValidationError(f"latitude must be a finite number, got {value!r}")
    if not (-90.0 <= float(value) <= 90.0):
        raise GpsValidationError(f"latitude {value} out of WGS84 range [-90, 90]")
    return float(value)


def validate_longitude(value: float) -> float:
    """Return ``value`` if it is a finite WGS84 longitude, else raise."""
    if not _is_finite(value):
        raise GpsValidationError(f"longitude must be a finite number, got {value!r}")
    if not (-180.0 <= float(value) <= 180.0):
        raise GpsValidationError(f"longitude {value} out of WGS84 range [-180, 180]")
    return float(value)


def validate_heading(value: float) -> float:
    """Normalise a heading to ``[0, 360)``; reject NaN/inf."""
    if not _is_finite(value):
        raise GpsValidationError(f"heading must be a finite number, got {value!r}")
    return float(value) % 360.0


def is_null_island(lat: float, lon: float, *, tolerance_deg: float = 1e-4) -> bool:
    """Detect the (0, 0) "Null Island" sentinel emitted by buggy GPS firmware.

    Real recordings within ~10 metres of (0, 0) are essentially impossible
    for a Romanian-deployed system, so we treat any reading inside the
    tolerance as a bad fix rather than a coincidence.
    """
    return abs(lat) < tolerance_deg and abs(lon) < tolerance_deg


def is_inside_romania(lat: float, lon: float) -> bool:
    """Bounding-box check against the Romanian ROI used by ParkGuard."""
    return (
        ROMANIA_LAT_MIN <= lat <= ROMANIA_LAT_MAX
        and ROMANIA_LON_MIN <= lon <= ROMANIA_LON_MAX
    )


def validate_coordinates(
    lat: float,
    lon: float,
    *,
    require_country: bool = True,
) -> ValidatedCoordinates:
    """Run every hard rule against a single ``(lat, lon)`` pair.

    Hard rules raise :class:`GpsValidationError`:
        1. NaN / inf / non-numeric.
        2. Out of WGS84 range.
        3. Null Island (~0, 0).
        4. Outside Romania (when ``require_country=True``).

    Returns a :class:`ValidatedCoordinates` on success — using the wrapper
    type rather than the bare tuple makes it obvious downstream that the
    pair has been checked, so we never re-validate or accidentally pass an
    unchecked tuple to the projector.
    """
    lat_ok = validate_latitude(lat)
    lon_ok = validate_longitude(lon)
    if is_null_island(lat_ok, lon_ok):
        raise GpsValidationError(
            f"GPS reads ({lat_ok}, {lon_ok}) — Null Island sentinel, "
            "almost certainly a bad fix"
        )
    if require_country and not is_inside_romania(lat_ok, lon_ok):
        raise GpsValidationError(
            f"GPS ({lat_ok}, {lon_ok}) outside Romania bounding box "
            f"({ROMANIA_LAT_MIN}–{ROMANIA_LAT_MAX} N, "
            f"{ROMANIA_LON_MIN}–{ROMANIA_LON_MAX} E)"
        )
    return ValidatedCoordinates(latitude=lat_ok, longitude=lon_ok)


def haversine_distance_m(
    a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Great-circle distance in metres between two ``(lat, lon)`` pairs."""
    return geodesic(a, b).meters


def cross_validate_sources(
    primary: ValidatedCoordinates,
    secondary: ValidatedCoordinates,
    *,
    tolerance_m: float = CROSS_SOURCE_TOLERANCE_M,
) -> tuple[bool, float]:
    """Return ``(agree, distance_m)`` for two independently-sourced fixes.

    ``agree`` is True when ``distance_m <= tolerance_m``.  Caller decides
    what to do on disagreement — the typical path is "trust the primary,
    record a warning" so the operator can re-check the upload pairing.
    """
    distance = haversine_distance_m(
        (primary.latitude, primary.longitude),
        (secondary.latitude, secondary.longitude),
    )
    return (distance <= tolerance_m, distance)


def collect_warnings(*messages: str | None) -> list[str]:
    """Drop ``None``/empty entries and return a list of human-readable warnings."""
    return [m for m in messages if m]


def summarise_status(warnings: Iterable[str]) -> GpsValidationStatus:
    """Map the count of soft warnings to a :class:`GpsValidationStatus`."""
    return (
        GpsValidationStatus.WARNING
        if any(True for _ in warnings)
        else GpsValidationStatus.OK
    )
