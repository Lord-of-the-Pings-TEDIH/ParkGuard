"""Spot-occupancy scoring for Mobile-LPR patrols.

Each time the pipeline finalises a WRONG_PLATE track at a reserved spot,
one event is logged per patrol session (duplicates within the same session
are silently skipped).  A suspicion score is then recomputed for that
(spot, plate) pair from all events in the last 30 days.

Score formula
-------------
    score = base_score + spread_bonus + duration_bonus

    base_score  = Σ time_weight × decay(days_ago)
                  summed over all events in the last 30 days

    time_weight = 1.5 if detected 20:00–08:00 or on a weekend, else 1.0

    decay(d)    = 2^(−d / DECAY_HALF_LIFE_DAYS)
                  half-life 15 days: an event from today counts fully,
                  one from 15 days ago counts as 0.5, 30 days ago as 0.25

    spread_bonus = min((distinct_sessions − 1) × 0.5, 2.5)
                   linear bonus for each additional patrol session that
                   caught the plate, capped at 2.5 (five sessions).
                   Uses sessions, not days, so back-to-back passes on the
                   same day each count.

    duration_bonus = per calendar day: min(gap_hours / 8, 1.5) when two
                     distinct patrol sessions visited the spot ≥ 3 h apart
                     on that day — evidence of long-term occupation.

Why these changes over the original formula
-------------------------------------------
* The original ``raw_sum × recurrence_weight`` was quadratic: more events
  raised both terms simultaneously, hitting 55+ for ten daily sightings.
  The new formula grows linearly in both dimensions.
* Time decay prevents old events from dominating the score indefinitely.
* Session deduplication ensures one patrol pass = one event, even if OCR
  splits the plate into two tracks inside the same video.

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

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.occupancy import SpotOccupancyEvent, SpotOccupancyRecord
from app.models.parking_spot import ParkingSpot

UTC = timezone.utc
FLAG_THRESHOLD: float = 8.0
_LOOKBACK_DAYS: int = 30
_DECAY_HALF_LIFE_DAYS: float = 15.0
_DECAY_K: float = math.log(2) / _DECAY_HALF_LIFE_DAYS


def _time_weight(detected_at: datetime) -> float:
    """1.5 for night (20:00–08:00) or weekend, 1.0 otherwise."""
    hour = detected_at.hour
    weekday = detected_at.weekday()  # 0=Mon … 6=Sun
    if weekday >= 5 or hour >= 20 or hour < 8:
        return 1.5
    return 1.0


def _decay(days_ago: float) -> float:
    """Exponential decay: full weight today, half at DECAY_HALF_LIFE_DAYS."""
    return math.exp(-_DECAY_K * max(0.0, days_ago))


async def _recompute_score(
    spot_id: int,
    plate_text: str,
    db: AsyncSession,
) -> tuple[float, int]:
    """Return (score, distinct_days) computed from events in the last 30 days."""
    cutoff = datetime.now(UTC) - timedelta(days=_LOOKBACK_DAYS)
    result = await db.execute(
        select(SpotOccupancyEvent)
        .where(SpotOccupancyEvent.spot_id == spot_id)
        .where(SpotOccupancyEvent.plate_text == plate_text)
        .where(SpotOccupancyEvent.detected_at >= cutoff)
        .order_by(SpotOccupancyEvent.detected_at)
    )
    events = result.scalars().all()

    if not events:
        return 0.0, 0

    now = datetime.now(UTC)

    # Base score: per-event time_weight attenuated by age.
    base_score = sum(
        e.time_weight * _decay((now - e.detected_at).total_seconds() / 86400)
        for e in events
    )

    distinct_days = len({e.detected_at.date() for e in events})

    # Spread bonus: each additional distinct patrol session adds 0.5, capped at 2.5.
    # Null session_ids are grouped together as one anonymous session.
    distinct_sessions = len({e.session_id for e in events})
    spread_bonus = min(max(0, distinct_sessions - 1) * 0.5, 2.5)

    score = base_score + spread_bonus

    # Duration bonus: for each calendar day where ≥2 patrol sessions visited
    # the same spot, add a bonus proportional to the gap (evidences all-day use).
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

    return score, distinct_days


async def record_occupancy_event(
    spot_id: int,
    plate_text: str,
    detected_at: datetime,
    detection_id: uuid.UUID | None,
    session_id: uuid.UUID | None,
    db: AsyncSession,
    flag_threshold: float = FLAG_THRESHOLD,
) -> None:
    """Log a foreign-plate occupancy event and refresh the suspicion record.

    If an event for this (spot, plate, session) already exists the call is a
    no-op — one patrol pass counts once regardless of how many tracks OCR
    extracted from the video.
    """
    # Session deduplication: one event per (spot, plate, session).
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

    score, distinct_days = await _recompute_score(spot_id, plate_text, db)

    result = await db.execute(
        select(SpotOccupancyRecord)
        .where(SpotOccupancyRecord.spot_id == spot_id)
        .where(SpotOccupancyRecord.plate_text == plate_text)
        .limit(1)
    )
    record = result.scalars().first()

    if record is None:
        db.add(
            SpotOccupancyRecord(
                spot_id=spot_id,
                plate_text=plate_text,
                event_count=1,
                distinct_days=distinct_days,
                accumulated_score=score,
                first_seen_at=detected_at,
                last_seen_at=detected_at,
                is_flagged=score >= flag_threshold,
            )
        )
    else:
        record.event_count += 1
        record.distinct_days = distinct_days
        record.accumulated_score = score
        record.last_seen_at = detected_at
        record.is_flagged = score >= flag_threshold

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

    Must be called BEFORE the Session row is deleted — the FK on
    SpotOccupancyEvent.session_id is SET NULL on cascade, so once the session
    is gone we can no longer identify which events belonged to it.

    For each (spot, plate) pair that loses events:
    - If no events remain at all → delete the record entirely.
    - Otherwise → recompute score/flag from the surviving events.
    """
    # Collect affected (spot_id, plate_text) pairs before deleting anything.
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
        # Count all remaining events (not just the 30-day window) to decide
        # whether to keep or delete the record.
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
            score, distinct_days = await _recompute_score(spot_id, plate_text, db)
            record.event_count = remaining
            record.distinct_days = distinct_days
            record.accumulated_score = score
            record.is_flagged = score >= FLAG_THRESHOLD

    await db.flush()
