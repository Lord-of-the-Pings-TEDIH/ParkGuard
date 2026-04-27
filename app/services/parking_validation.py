"""Spatial validation for Mobile-LPR detections.

Given the GPS coordinate produced by ``PlateGeolocationCalculator`` and the
plate text read by OCR, decide whether the vehicle is parked on a spot it is
allowed to occupy.

The lookup uses a latitude/longitude bounding-box prefilter in SQL (cheap,
backed by ``ix_parking_spots_lat_lon``) and a Haversine refinement in
Python on the small candidate set.  This avoids a hard dependency on
PostGIS while staying accurate at the metre scale we care about.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.parking_spot import ParkingSpot


SpotMatchStatus = Literal["MATCH", "WRONG_PLATE", "NO_SPOT_FOUND"]

DEFAULT_SEARCH_RADIUS_M: float = 40.0
EARTH_RADIUS_M: float = 6_371_000.0
_METERS_PER_DEGREE_LAT: float = 111_320.0

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


def _canonical_plate_text(plate_text: str) -> str:
    """Match the canonicalisation used by ``app.services.ticket_lookup``."""
    return _NON_ALNUM_RE.sub("", plate_text.upper())


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class SpotMatch:
    status: SpotMatchStatus
    spot: ParkingSpot | None
    distance_m: float | None


async def validate_parking_at_location(
    target_lat: float,
    target_lon: float,
    plate_text: str,
    db: AsyncSession,
    radius_m: float = DEFAULT_SEARCH_RADIUS_M,
) -> SpotMatch:
    """Decide whether ``plate_text`` is parked on the right ``ParkingSpot``.

    Returns
    -------
    ``SpotMatch.status``:
      - ``"MATCH"``        – nearest spot's ``assigned_plate`` equals ``plate_text``
      - ``"WRONG_PLATE"``  – a spot is in range but assigned to someone else
      - ``"NO_SPOT_FOUND"``– no registered spot within ``radius_m``
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")

    # Convert the metres-radius into a latitude/longitude window so the SQL
    # prefilter can use a B-tree range scan instead of computing trig per row.
    cos_lat = max(math.cos(math.radians(target_lat)), 1e-6)
    delta_lat = radius_m / _METERS_PER_DEGREE_LAT
    delta_lon = radius_m / (_METERS_PER_DEGREE_LAT * cos_lat)

    # Only consider reserved spots (assigned_plate IS NOT NULL).  An unreserved
    # spot has no owner to wrong, so it must not trigger a WRONG_PLATE result
    # or contribute to the occupancy-scoring pipeline.
    stmt = select(ParkingSpot).where(
        ParkingSpot.latitude.is_not(None),
        ParkingSpot.longitude.is_not(None),
        ParkingSpot.assigned_plate.is_not(None),
        ParkingSpot.latitude.between(target_lat - delta_lat, target_lat + delta_lat),
        ParkingSpot.longitude.between(target_lon - delta_lon, target_lon + delta_lon),
    )

    result = await db.execute(stmt)
    candidates = result.scalars().all()

    nearest: ParkingSpot | None = None
    nearest_distance: float | None = None
    for spot in candidates:
        # latitude / longitude are non-NULL by the query above; assert for type
        # narrowing without asserting at runtime.
        assert spot.latitude is not None and spot.longitude is not None
        d = _haversine_m(target_lat, target_lon, spot.latitude, spot.longitude)
        if d <= radius_m and (nearest_distance is None or d < nearest_distance):
            nearest = spot
            nearest_distance = d

    if nearest is None:
        return SpotMatch(status="NO_SPOT_FOUND", spot=None, distance_m=None)

    # nearest.assigned_plate is guaranteed non-NULL by the query filter above.
    expected = _canonical_plate_text(nearest.assigned_plate)  # type: ignore[arg-type]
    actual = _canonical_plate_text(plate_text)

    status: SpotMatchStatus = "MATCH" if expected == actual else "WRONG_PLATE"
    return SpotMatch(status=status, spot=nearest, distance_m=nearest_distance)
