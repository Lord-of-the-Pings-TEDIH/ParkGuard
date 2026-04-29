"""Spot-occupancy scoring for Mobile-LPR patrols.

Each time the pipeline finalises a WRONG_PLATE track at a reserved spot,
one event is logged per patrol session (duplicates within the same session
are silently skipped).  A suspicion score is then recomputed for that
(spot, plate) pair from all events in the last OCCUPANCY_LOOKBACK_DAYS days.

Score formula
-------------
    score = base_score + spread_bonus + duration_bonus

    base_score  = Σ time_weight × decay(days_ago)
                  summed over all events in the lookback window

    time_weight = 1.5 if detected 20:00–08:00 or on a weekend, else 1.0

    decay(d)    = 2^(−d / DECAY_HALF_LIFE_DAYS)

    spread_bonus = min((distinct_sessions − 1) × 0.5, 2.5)
                   linear bonus for each additional patrol session that
                   caught the plate, capped at 2.5 (five sessions).

    duration_bonus = per calendar day: min(gap_hours / 8, 1.5) when two
                     distinct patrol sessions visited the spot ≥ 3 h apart
                     on that day.

Hour concentration
------------------
    hour_concentration ∈ [0, 1] measures how consistently events cluster at
    the same time of day.  1 = all events at exactly the same hour,
    0 = events spread uniformly over 24 hours.  Darius specifically
    mentions "la ore similare (de exemplu 3 noaptea)" as a red flag.

Score reset
-----------
When the assigned owner's plate is detected at their spot the accumulated
score and flag are cleared.  Raw events are kept for audit.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.occupancy import SpotOccupancyEvent, SpotOccupancyRecord
from app.models.parking_spot import ParkingSpot

UTC = timezone.utc
_RO_TZ = ZoneInfo("Europe/Bucharest")


def _ro_hour(dt: datetime) -> int:
    """Return the hour-of-day in Romanian local time (handles DST automatically)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_RO_TZ).hour

# Read thresholds from settings so operators can tune without code changes.
FLAG_THRESHOLD: float = settings.OCCUPANCY_FLAG_THRESHOLD
_LOOKBACK_DAYS: int = settings.OCCUPANCY_LOOKBACK_DAYS
_DECAY_HALF_LIFE_DAYS: float = settings.OCCUPANCY_DECAY_HALF_LIFE_DAYS
_DECAY_K: float = math.log(2) / _DECAY_HALF_LIFE_DAYS
_HOUR_CONC_THRESHOLD: float = settings.OCCUPANCY_HOUR_CONCENTRATION_THRESHOLD
_HOUR_CONC_MIN_EVENTS: int = settings.OCCUPANCY_HOUR_CONCENTRATION_MIN_EVENTS
_HOUR_CONC_MULTIPLIER: float = settings.OCCUPANCY_HOUR_CONCENTRATION_MULTIPLIER
_MIN_EVENTS_FOR_FLAG: int = settings.OCCUPANCY_MIN_EVENTS_FOR_FLAG


def _should_flag(score: float, event_count: int, threshold: float = FLAG_THRESHOLD) -> bool:
    """Return True when score AND event_count both clear their thresholds."""
    if score < threshold:
        return False
    if _MIN_EVENTS_FOR_FLAG > 0 and event_count < _MIN_EVENTS_FOR_FLAG:
        return False
    return True


def _time_weight(detected_at: datetime) -> float:
    """1.5 for night (20:00–08:00) or weekend in Romanian local time, 1.0 otherwise."""
    if detected_at.tzinfo is None:
        detected_at = detected_at.replace(tzinfo=UTC)
    local_dt = detected_at.astimezone(_RO_TZ)
    hour = local_dt.hour
    weekday = local_dt.weekday()  # 0=Mon … 6=Sun
    if weekday >= 5 or hour >= 20 or hour < 8:
        return 1.5
    return 1.0


def _decay(days_ago: float) -> float:
    """Exponential decay: full weight today, half at DECAY_HALF_LIFE_DAYS."""
    return math.exp(-_DECAY_K * max(0.0, days_ago))


def _compute_hour_concentration(events: list[SpotOccupancyEvent]) -> float | None:
    """Measure how consistently events cluster at the same hour of day.

    Returns a value in [0, 1]:
      1.0 = all events at the same hour (maximally concentrated)
      0.0 = events uniformly spread across all 24 hours

    Uses circular statistics for hours (wraps around midnight so
    22:00 and 02:00 are recognised as close together).

    Returns None when fewer than 2 events are present.
    """
    if len(events) < 2:
        return None

    # Map each hour to an angle on the unit circle (0h = 0°, 23h = 345°).
    angles = [_ro_hour(e.detected_at) * (2 * math.pi / 24) for e in events]
    n = len(angles)
    mean_sin = sum(math.sin(a) for a in angles) / n
    mean_cos = sum(math.cos(a) for a in angles) / n
    # Resultant length R: 1 = perfectly concentrated, 0 = maximally dispersed.
    return math.sqrt(mean_sin ** 2 + mean_cos ** 2)


def _compute_typical_hour(events: list[SpotOccupancyEvent]) -> float | None:
    """Return the circular mean hour of day in Romanian local time (0–24), or None."""
    if not events:
        return None
    angles = [_ro_hour(e.detected_at) * (2 * math.pi / 24) for e in events]
    n = len(angles)
    mean_sin = sum(math.sin(a) for a in angles) / n
    mean_cos = sum(math.cos(a) for a in angles) / n
    angle = math.atan2(mean_sin, mean_cos)
    hour = (angle * 24 / (2 * math.pi)) % 24
    return hour


async def _recompute_score(
    spot_id: int,
    plate_text: str,
    db: AsyncSession,
) -> tuple[float, int, float | None, float | None]:
    """Return (score, distinct_days, hour_concentration, typical_hour)."""
    # Use a naive UTC cutoff for the DB query so that both SQLite (which
    # stores timestamps without timezone) and PostgreSQL work correctly.
    # PostgreSQL accepts naive datetimes and treats them as UTC when the
    # session timezone is UTC (the asyncpg default).
    cutoff_naive = datetime.utcnow() - timedelta(days=_LOOKBACK_DAYS)
    result = await db.execute(
        select(SpotOccupancyEvent)
        .where(SpotOccupancyEvent.spot_id == spot_id)
        .where(SpotOccupancyEvent.plate_text == plate_text)
        .where(SpotOccupancyEvent.detected_at >= cutoff_naive)
        .order_by(SpotOccupancyEvent.detected_at)
    )
    events = result.scalars().all()

    if not events:
        return 0.0, 0, None, None

    now = datetime.now(UTC)

    def _as_aware(dt: datetime) -> datetime:
        # SQLite returns naive datetimes; treat them as UTC so tests work.
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

    base_score = sum(
        e.time_weight * _decay((now - _as_aware(e.detected_at)).total_seconds() / 86400)
        for e in events
    )

    # Apply hour-concentration multiplier when events cluster at a consistent
    # time of day and there are enough of them to be statistically meaningful.
    hour_conc_early = _compute_hour_concentration(list(events))
    if (
        hour_conc_early is not None
        and len(events) >= _HOUR_CONC_MIN_EVENTS
        and hour_conc_early >= _HOUR_CONC_THRESHOLD
    ):
        base_score *= _HOUR_CONC_MULTIPLIER

    distinct_days = len({e.detected_at.date() for e in events})

    distinct_sessions = len({e.session_id for e in events})
    spread_bonus = min(max(0, distinct_sessions - 1) * 0.5, 2.5)

    score = base_score + spread_bonus

    by_day: dict[object, dict[uuid.UUID | None, datetime]] = defaultdict(dict)
    for e in events:
        day = e.detected_at.date()
        sid = e.session_id
        if sid not in by_day[day] or e.detected_at < by_day[day][sid]:
            by_day[day][sid] = e.detected_at

    for sessions_on_day in by_day.values():
        if len(sessions_on_day) >= 2:
            timestamps = sorted(sessions_on_day.values())
            gap_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            if gap_hours >= 3:
                score += min(gap_hours / 8.0, 1.5)

    hour_conc = _compute_hour_concentration(list(events))
    typical_hour = _compute_typical_hour(list(events))

    return score, distinct_days, hour_conc, typical_hour


async def record_occupancy_event(
    spot_id: int,
    plate_text: str,
    detected_at: datetime,
    detection_id: uuid.UUID | None,
    session_id: uuid.UUID | None,
    db: AsyncSession,
    flag_threshold: float = FLAG_THRESHOLD,
    owner_spot: ParkingSpot | None = None,
) -> None:
    """Log a foreign-plate occupancy event and refresh the suspicion record.

    If an event for this (spot, plate, session) already exists the call is a
    no-op — one patrol pass counts once regardless of how many tracks OCR
    extracted from the video.  When the record transitions from unflagged to
    flagged an Alert is created automatically.
    """
    if session_id is not None:
        dup = await db.execute(
            select(SpotOccupancyEvent.id)
            .where(SpotOccupancyEvent.spot_id == spot_id)
            .where(SpotOccupancyEvent.plate_text == plate_text)
            .where(SpotOccupancyEvent.session_id == session_id)
            .limit(1)
        )
        if dup.scalars().first() is not None:
            return

    db.add(
        SpotOccupancyEvent(
            spot_id=spot_id,
            plate_text=plate_text,
            detected_at=detected_at,
            detection_id=detection_id,
            session_id=session_id,
            time_weight=_time_weight(detected_at),
        )
    )
    await db.flush()

    score, distinct_days, hour_conc, typical_hour = await _recompute_score(
        spot_id, plate_text, db
    )

    result = await db.execute(
        select(SpotOccupancyRecord)
        .where(SpotOccupancyRecord.spot_id == spot_id)
        .where(SpotOccupancyRecord.plate_text == plate_text)
        .limit(1)
    )
    record = result.scalars().first()

    previously_flagged = record.is_flagged if record is not None else False
    new_event_count = (record.event_count + 1) if record is not None else 1
    now_flagged = _should_flag(score, new_event_count, flag_threshold)

    if record is None:
        db.add(
            SpotOccupancyRecord(
                spot_id=spot_id,
                plate_text=plate_text,
                event_count=1,
                distinct_days=distinct_days,
                accumulated_score=score,
                hour_concentration=hour_conc,
                typical_hour=typical_hour,
                first_seen_at=detected_at,
                last_seen_at=detected_at,
                is_flagged=now_flagged,
            )
        )
    else:
        record.event_count += 1
        record.distinct_days = distinct_days
        record.accumulated_score = score
        record.hour_concentration = hour_conc
        record.typical_hour = typical_hour
        record.last_seen_at = detected_at
        record.is_flagged = now_flagged

    await db.flush()

    # Create an alert the first time a record crosses the flag threshold.
    if now_flagged and not previously_flagged:
        from app.models.alert import Alert
        spot_result = await db.execute(
            select(ParkingSpot).where(ParkingSpot.id == spot_id).limit(1)
        )
        spot = spot_result.scalars().first()
        spot_label = spot.spot_label if spot else str(spot_id)
        hour_info = (
            f" (ora tipică: {typical_hour:.0f}h)" if typical_hour is not None else ""
        )
        neighbour_info = ""
        if (
            owner_spot is not None
            and owner_spot.spot_sequence is not None
            and spot is not None
            and spot.spot_sequence is not None
        ):
            delta = owner_spot.spot_sequence - spot.spot_sequence
            neighbour_info = (
                f" (proprietar: {owner_spot.spot_label}, Δseq={delta:+d})"
            )
        db.add(
            Alert(
                detection_id=detection_id,
                alert_type="spot_misuse",
                message=(
                    f"Plăcuța {plate_text} detectată de {record.event_count if record else 1}× "  # type: ignore[union-attr]
                    f"pe locul {spot_label}{hour_info}{neighbour_info}"
                ),
                is_resolved=False,
            )
        )
        await db.flush()


async def reset_spot_suspicion(spot_id: int, db: AsyncSession) -> None:
    """Clear score and flag for all foreign-plate records at this spot.

    Called when the assigned owner's plate is detected, signalling they
    are actively using the spot.  Raw events are preserved for audit.
    """
    await db.execute(
        update(SpotOccupancyRecord)
        .where(SpotOccupancyRecord.spot_id == spot_id)
        .values(accumulated_score=0.0, is_flagged=False)
    )
    await db.flush()


@dataclass
class EnrichedOccupancyRecord:
    id: int
    spot_id: int
    spot_label: str | None
    assigned_plate: str | None
    plate_text: str
    event_count: int
    distinct_days: int
    accumulated_score: float
    hour_concentration: float | None
    typical_hour: float | None
    first_seen_at: datetime
    last_seen_at: datetime
    is_flagged: bool


async def get_suspicious_records(
    db: AsyncSession,
    flagged_only: bool = True,
) -> list[EnrichedOccupancyRecord]:
    """Return occupancy records sorted by score descending, enriched with spot info."""
    stmt = select(SpotOccupancyRecord)
    if flagged_only:
        stmt = stmt.where(SpotOccupancyRecord.is_flagged.is_(True))
    stmt = stmt.order_by(SpotOccupancyRecord.accumulated_score.desc())
    records = list((await db.execute(stmt)).scalars().all())

    if not records:
        return []

    spot_ids = {r.spot_id for r in records}
    spots = {
        s.id: s
        for s in (
            await db.execute(select(ParkingSpot).where(ParkingSpot.id.in_(spot_ids)))
        ).scalars().all()
    }

    return [
        EnrichedOccupancyRecord(
            id=r.id,
            spot_id=r.spot_id,
            spot_label=spots[r.spot_id].spot_label if r.spot_id in spots else None,
            assigned_plate=spots[r.spot_id].assigned_plate if r.spot_id in spots else None,
            plate_text=r.plate_text,
            event_count=r.event_count,
            distinct_days=r.distinct_days,
            accumulated_score=r.accumulated_score,
            hour_concentration=getattr(r, "hour_concentration", None),
            typical_hour=getattr(r, "typical_hour", None),
            first_seen_at=r.first_seen_at,
            last_seen_at=r.last_seen_at,
            is_flagged=r.is_flagged,
        )
        for r in records
    ]


async def reset_occupancy_record(record_id: int, db: AsyncSession) -> bool:
    """Clear score and flag for a specific record. Returns False if not found."""
    result = await db.execute(
        select(SpotOccupancyRecord).where(SpotOccupancyRecord.id == record_id).limit(1)
    )
    record = result.scalars().first()
    if record is None:
        return False
    record.accumulated_score = 0.0
    record.is_flagged = False
    await db.flush()
    return True


async def cleanup_session_occupancy(session_id: uuid.UUID, db: AsyncSession) -> None:
    """Remove occupancy events tied to a session and recompute affected records.

    Must be called BEFORE the Session row is deleted.
    """
    pairs_result = await db.execute(
        select(SpotOccupancyEvent.spot_id, SpotOccupancyEvent.plate_text)
        .where(SpotOccupancyEvent.session_id == session_id)
        .distinct()
    )
    affected_pairs = pairs_result.all()

    if not affected_pairs:
        return

    await db.execute(
        delete(SpotOccupancyEvent).where(SpotOccupancyEvent.session_id == session_id)
    )
    await db.flush()

    for spot_id, plate_text in affected_pairs:
        count_result = await db.execute(
            select(func.count(SpotOccupancyEvent.id))
            .where(SpotOccupancyEvent.spot_id == spot_id)
            .where(SpotOccupancyEvent.plate_text == plate_text)
        )
        remaining = count_result.scalar_one()

        record_result = await db.execute(
            select(SpotOccupancyRecord)
            .where(SpotOccupancyRecord.spot_id == spot_id)
            .where(SpotOccupancyRecord.plate_text == plate_text)
            .limit(1)
        )
        record = record_result.scalars().first()
        if record is None:
            continue

        if remaining == 0:
            await db.delete(record)
        else:
            score, distinct_days, hour_conc, typical_hour = await _recompute_score(
                spot_id, plate_text, db
            )
            record.event_count = remaining
            record.distinct_days = distinct_days
            record.accumulated_score = score
            record.hour_concentration = hour_conc
            record.typical_hour = typical_hour
            record.is_flagged = _should_flag(score, remaining)

    await db.flush()
