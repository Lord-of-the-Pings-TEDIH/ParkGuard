"""Simulate residential spot-misuse from real video1.MOV detections.

The detection pipeline already projects every plate to a GPS coordinate and,
when a registered ParkingSpot sits within ``MOBILE_LPR_SEARCH_RADIUS_M``,
calls ``record_occupancy_event`` with a ``WRONG_PLATE`` verdict.  That keeps
working for live video.  This script is the *demo* path: it backfills a
realistic spread of suspicion scores for the plates that already exist in
the database, so the Spot Misuse panel has data to show without re-running
the full pipeline.

Two design choices worth flagging:

1. ``session_id`` on the synthesised events is **NULL** on purpose.  The
   real pipeline sets it to the patrol session that produced the read so
   ``cleanup_session_occupancy`` can drop the events when that session is
   deleted.  For demo data we want the opposite: deleting the original
   video1 session must NOT erase the simulated patrol history, so the
   panel stays populated across DB cleanups.

2. Each plate is bound to its closest registered spot using the actual
   per-detection projected coords (``target_latitude``/``target_longitude``)
   stored by the live pipeline — same haversine + radius logic the runtime
   validator uses.  Multiple events per plate are spread across different
   calendar days and evening hours so the time-decay + spread bonuses in
   ``_recompute_score`` produce a believable distribution of scores.

Run from the project root after the API has booted at least once (so the
residential lot from ``app/core/seed.py`` exists):

    venv/bin/python seed/seed_residential_misuse.py
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.seed import _seed_residential_lot
from app.models.detection import Detection, Frame
from app.models.occupancy import SpotOccupancyEvent, SpotOccupancyRecord
from app.models.parking_spot import ParkingSpot
from app.services.occupancy import FLAG_THRESHOLD, _recompute_score, _time_weight
from app.services.parking_validation import (
    _canonical_plate_text,
    validate_parking_at_location,
)

UTC = timezone.utc

# Per-plate event count.  Range (min, max) drawn uniformly so we end up with
# a mix of scores: a plate with 3 events in 3 days lands below the flag
# threshold; one with 8 events across a week clears it comfortably.  Keeps
# the panel showing both flagged and below-threshold rows when the user
# toggles "Flagged only" off.
_EVENTS_PER_PLATE_RANGE = (3, 9)
# Spread events across this many days.  Overlap with _EVENTS_PER_PLATE_RANGE
# is intentional — a plate with 8 events in 7 days will see at least two
# events on the same day, which triggers the duration_bonus in the scorer.
_LOOKBACK_DAYS = 8
_EVENING_HOURS = (20, 21, 22, 23)


def _earth_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine in metres — duplicated locally to avoid importing the private helper."""
    earth_r = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * earth_r * math.asin(math.sqrt(a))


async def _load_residential_spots(db) -> list[ParkingSpot]:
    """Reserved spots only — unreserved ones can't host a WRONG_PLATE verdict."""
    rows = (
        await db.execute(
            select(ParkingSpot).where(
                ParkingSpot.assigned_plate.is_not(None),
                ParkingSpot.latitude.is_not(None),
                ParkingSpot.longitude.is_not(None),
            )
        )
    ).scalars().all()
    return list(rows)


async def _load_video1_plates(
    db,
) -> list[tuple[str, float, float]]:
    """Return ``(canonical_plate, avg_lat, avg_lon)`` for each real video1 plate.

    Averaging the projected coords across all frames of the same plate gives
    a stable per-plate position that matches what an operator would see on
    the map view — independent of OCR reading order.
    """
    rows = (
        await db.execute(
            select(
                Detection.ocr_normalized_text,
                Detection.target_latitude,
                Detection.target_longitude,
            )
            .join(Frame, Frame.id == Detection.frame_id)
            .where(Detection.is_valid_ro_plate.is_(True))
            .where(Detection.target_latitude.is_not(None))
            .where(Detection.target_longitude.is_not(None))
        )
    ).all()

    grouped: dict[str, list[tuple[float, float]]] = {}
    for plate_text, lat, lon in rows:
        if not plate_text:
            continue
        canonical = _canonical_plate_text(plate_text)
        if not canonical:
            continue
        grouped.setdefault(canonical, []).append((float(lat), float(lon)))

    return [
        (
            plate,
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
        for plate, points in grouped.items()
    ]


def _nearest_spot(
    plate_lat: float,
    plate_lon: float,
    spots: list[ParkingSpot],
) -> ParkingSpot | None:
    """Return the spot closest to ``(plate_lat, plate_lon)`` within search radius."""
    if not spots:
        return None
    radius = float(settings.MOBILE_LPR_SEARCH_RADIUS_M)
    best: ParkingSpot | None = None
    best_d = float("inf")
    for spot in spots:
        d = _earth_distance_m(plate_lat, plate_lon, spot.latitude, spot.longitude)
        if d <= radius and d < best_d:
            best = spot
            best_d = d
    return best


def _generate_event_timestamps(now: datetime, count: int) -> list[datetime]:
    """Spread ``count`` evening timestamps across the lookback window.

    Picks distinct (day_offset, hour) pairs so we don't synthesise duplicate
    events (which would survive the session-dedup check anyway, but produce
    visually identical rows).  Falls back to 'with replacement' once the
    pool is exhausted — happens only when count > _LOOKBACK_DAYS * len(_EVENING_HOURS).
    """
    pool = [
        (day, hour)
        for day in range(_LOOKBACK_DAYS)
        for hour in _EVENING_HOURS
    ]
    random.shuffle(pool)
    chosen = pool[:count] if count <= len(pool) else pool + random.choices(pool, k=count - len(pool))
    return [
        (now - timedelta(days=day)).replace(hour=hour, minute=random.randint(0, 59), second=0, microsecond=0)
        for day, hour in chosen
    ]


async def main() -> None:
    random.seed(2026_04_27)  # deterministic demo data
    async with AsyncSessionLocal() as db:
        # Make the script self-bootstrapping: if the residential lot has
        # been wiped (or never created), recreate it from the canonical
        # _DEMO_SPOTS list before populating misuse data.
        await _seed_residential_lot(db)

        spots = await _load_residential_spots(db)
        if not spots:
            print(
                "✗ No reserved parking spots found after seed.  Check that "
                "app/core/seed.py defines _DEMO_SPOTS with assigned plates."
            )
            return

        plates = await _load_video1_plates(db)
        if not plates:
            print(
                "✗ No projected detections found.  Process video1.MOV with "
                "GPS pose set on the session, then re-run."
            )
            return

        # Wipe any prior synthesised data so reruns are idempotent.  Both
        # tables cascade-delete from spot_id, but we also want to clear
        # records that point at spots which still exist.
        await db.execute(delete(SpotOccupancyRecord))
        await db.execute(delete(SpotOccupancyEvent))
        await db.commit()
        print(f"Wiped occupancy tables.  Spots available: {len(spots)}.")

        now = datetime.now(UTC)
        skipped: list[str] = []
        seeded: list[tuple[str, str, float, int, bool]] = []

        for plate_text, plate_lat, plate_lon in sorted(plates):
            spot = _nearest_spot(plate_lat, plate_lon, spots)
            if spot is None:
                skipped.append(f"{plate_text} (out of {settings.MOBILE_LPR_SEARCH_RADIUS_M:.0f} m)")
                continue

            # Sanity guard: a plate that legitimately matches its closest
            # spot is not a misuse case.  In our demo spots are owned by
            # invented residents (none in video1), so this should never
            # trigger — but it makes the script reusable on real data.
            if _canonical_plate_text(spot.assigned_plate or "") == plate_text:
                skipped.append(f"{plate_text} (matches owner of {spot.spot_label})")
                continue

            n_events = random.randint(*_EVENTS_PER_PLATE_RANGE)
            timestamps = sorted(_generate_event_timestamps(now, n_events))

            first_seen = timestamps[0]
            last_seen = timestamps[-1]

            for ts in timestamps:
                db.add(
                    SpotOccupancyEvent(
                        spot_id=spot.id,
                        plate_text=plate_text,
                        # Detached from any session: events survive the
                        # original video1 session being deleted, which is the
                        # whole point of the cross-session demo.
                        detection_id=None,
                        session_id=None,
                        detected_at=ts,
                        time_weight=_time_weight(ts),
                    )
                )

            await db.flush()

            score, distinct_days = await _recompute_score(spot.id, plate_text, db)
            db.add(
                SpotOccupancyRecord(
                    spot_id=spot.id,
                    plate_text=plate_text,
                    event_count=n_events,
                    distinct_days=distinct_days,
                    accumulated_score=score,
                    first_seen_at=first_seen,
                    last_seen_at=last_seen,
                    is_flagged=score >= FLAG_THRESHOLD,
                )
            )
            await db.commit()

            seeded.append(
                (plate_text, spot.spot_label, score, n_events, score >= FLAG_THRESHOLD)
            )

        # Print a tidy summary so the operator can spot-check the spread.
        print()
        print(f"{'Plate':<10s} {'Spot':<6s} {'Owner':<11s} {'Score':>6s} {'Events':>7s} {'Flag':>5s}")
        print("─" * 52)
        owner_by_spot = {s.spot_label: s.assigned_plate for s in spots}
        for plate_text, spot_label, score, n_events, flagged in sorted(
            seeded, key=lambda r: -r[2]
        ):
            print(
                f"{plate_text:<10s} {spot_label:<6s} "
                f"{(owner_by_spot.get(spot_label) or '—'):<11s} "
                f"{score:>6.2f} {n_events:>7d} {'★' if flagged else ' ':>5s}"
            )

        if skipped:
            print()
            print("Skipped:")
            for line in skipped:
                print(f"  - {line}")

        print()
        flagged_total = sum(1 for r in seeded if r[4])
        print(
            f"Done — {len(seeded)} (plate, spot) pairs seeded, "
            f"{flagged_total} flagged at score ≥ {FLAG_THRESHOLD}."
        )

        # ── Per-detection backfill ─────────────────────────────────────────
        # The existing video1 detections were processed when the only
        # registered spots were in Cluj-Napoca, so every row was stamped
        # ``NO_SPOT_FOUND``.  Now that the residential lot exists at the
        # right coordinates, replay validate_parking_at_location for each
        # detection with projected GPS so the per-card badges in the live
        # session view show WRONG_PLATE instead of NO_SPOT_FOUND.  This is
        # purely cosmetic (the suspicion score above is the load-bearing
        # data) but keeps the UI consistent with the new spot geometry.
        await _backfill_detection_spot_status(db)


async def _backfill_detection_spot_status(db) -> None:
    rows = (
        await db.execute(
            select(Detection)
            .where(Detection.target_latitude.is_not(None))
            .where(Detection.target_longitude.is_not(None))
            .where(Detection.is_valid_ro_plate.is_(True))
            .where(Detection.ocr_normalized_text.is_not(None))
        )
    ).scalars().all()

    if not rows:
        return

    updated = 0
    for det in rows:
        plate_canonical = _canonical_plate_text(det.ocr_normalized_text or "")
        if not plate_canonical:
            continue
        match = await validate_parking_at_location(
            target_lat=float(det.target_latitude),
            target_lon=float(det.target_longitude),
            plate_text=plate_canonical,
            db=db,
        )
        det.spot_match_status = match.status
        det.target_distance_m = match.distance_m
        det.matched_spot_id = match.spot.id if match.spot is not None else None
        updated += 1

    await db.commit()
    print()
    print(f"Backfilled {updated} detection rows with refreshed spot match status.")


if __name__ == "__main__":
    asyncio.run(main())
