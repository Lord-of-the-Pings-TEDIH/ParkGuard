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
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.parking_spot import ParkingSpot
from app.pipeline.plate_validator import compact_plate


SpotMatchStatus = Literal["MATCH", "WRONG_PLATE", "NO_SPOT_FOUND"]

DEFAULT_SEARCH_RADIUS_M: float = 40.0
EARTH_RADIUS_M: float = 6_371_000.0
_METERS_PER_DEGREE_LAT: float = 111_320.0


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
    owner_spot: ParkingSpot | None = field(default=None)


async def find_owner_spot(
    plate_text: str,
    parking_lot_id: int,
    exclude_spot_id: int,
    db: AsyncSession,
) -> ParkingSpot | None:
    """Find the spot registered to ``plate_text`` in the same lot.

    Used to surface the X / X+1 neighbour relationship when a foreign plate
    occupies a reserved spot: if ``plate_text`` itself owns spot N in the same
    lot, the alert message can note the sequence delta.
    """
    canonical = compact_plate(plate_text)
    result = await db.execute(
        select(ParkingSpot).where(
            ParkingSpot.parking_lot_id == parking_lot_id,
            ParkingSpot.assigned_plate == canonical,
            ParkingSpot.id != exclude_spot_id,
        ).limit(1)
    )
    return result.scalars().first()


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
      - ``"MATCH"``        – nearest spot's ``assigned_plate`` or ``allowed_plates`` matches
      - ``"WRONG_PLATE"``  – a spot is in range but assigned to someone else
      - ``"NO_SPOT_FOUND"``– no registered spot within ``radius_m``
    """
    if radius_m <= 0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")

    cos_lat = max(math.cos(math.radians(target_lat)), 1e-6)
    delta_lat = radius_m / _METERS_PER_DEGREE_LAT
    delta_lon = radius_m / (_METERS_PER_DEGREE_LAT * cos_lat)

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
        assert spot.latitude is not None and spot.longitude is not None
        d = _haversine_m(target_lat, target_lon, spot.latitude, spot.longitude)
        if d <= radius_m and (nearest_distance is None or d < nearest_distance):
            nearest = spot
            nearest_distance = d

    if nearest is None:
        return SpotMatch(status="NO_SPOT_FOUND", spot=None, distance_m=None)

    actual = compact_plate(plate_text)
    expected_owner = compact_plate(nearest.assigned_plate)  # type: ignore[arg-type]
    allowed = {compact_plate(p) for p in (nearest.allowed_plates or [])}

    if actual == expected_owner or actual in allowed:
        return SpotMatch(status="MATCH", spot=nearest, distance_m=nearest_distance)

    # WRONG_PLATE — look up whether the interloper owns a spot in the same lot
    owner_spot = await find_owner_spot(
        plate_text=actual,
        parking_lot_id=nearest.parking_lot_id,
        exclude_spot_id=nearest.id,
        db=db,
    )
    return SpotMatch(
        status="WRONG_PLATE",
        spot=nearest,
        distance_m=nearest_distance,
        owner_spot=owner_spot,
    )
