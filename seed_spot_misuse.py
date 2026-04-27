"""Simulate spot-misuse data using the real detections from video1.mov.

Reuses the existing sessions and detection rows in the database — no dummy
sessions are created. To produce a useful spread of scores, ``detected_at``
on each occupancy event is backdated across the past week, simulating a
plate being seen on multiple patrol passes (the actual video was processed
in one burst, so all real detection timestamps cluster on a single day).

Run from the project root:

    venv/bin/python seed_spot_misuse.py
"""

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.core.database import AsyncSessionLocal
from app.models.detection import Detection, Frame
from app.models.occupancy import SpotOccupancyEvent, SpotOccupancyRecord
from app.models.parking_spot import ParkingSpot
from app.services.occupancy import FLAG_THRESHOLD, _recompute_score, _time_weight

UTC = timezone.utc


# Spot misuse scenarios — (spot_label, intruder_plate_normalized, day_offsets)
# day_offsets describes how far back each simulated patrol pass was, so we
# can produce realistic distinct_days / decay even though the real video
# was processed all at once.
SCENARIOS = [
    # High flagged (top of list)
    ("A-01", "SJ 07 SMS", [0, 1, 2, 3, 4, 5, 6, 7]),
    # Mid-high flagged
    ("A-02", "SJ 05 DBA", [0, 1, 2, 4, 5, 6]),
    # Just over threshold
    ("A-03", "SJ 09 CRN", [0, 1, 2, 3, 5, 7]),
    # Below threshold (toggle "Flagged only" off to see)
    ("A-01", "B 775 SUM", [0, 2, 4]),
    ("A-02", "SJ 11 REM", [0, 1]),
]


async def main() -> None:
    async with AsyncSessionLocal() as db:
        spots = (await db.execute(select(ParkingSpot))).scalars().all()
        spots_by_label = {s.spot_label: s for s in spots}

        await db.execute(delete(SpotOccupancyRecord))
        await db.execute(delete(SpotOccupancyEvent))
        await db.commit()
        print("Wiped existing occupancy data.")

        now = datetime.now(UTC)

        for spot_label, plate_ocr, day_offsets in SCENARIOS:
            spot = spots_by_label.get(spot_label)
            if spot is None:
                print(f"  skip — spot {spot_label} not found")
                continue

            # Pull real detections of this plate (with their session_id via frame).
            rows = (await db.execute(
                select(Detection, Frame.session_id)
                .join(Frame, Frame.id == Detection.frame_id)
                .where(Detection.ocr_normalized_text == plate_ocr)
                .where(Detection.is_valid_ro_plate.is_(True))
            )).all()

            if not rows:
                print(f"  skip — no real detections for {plate_ocr}")
                continue

            # Compact form (matches what the live processor stores).
            plate_text = plate_ocr.replace(" ", "")

            first_seen: datetime | None = None
            last_seen: datetime | None = None

            for i, days_ago in enumerate(day_offsets):
                detection, session_id = rows[i % len(rows)]
                # Evening / night times so weight = 1.5 (typical for misuse).
                hour = 21 + (i % 3)  # 21, 22, 23 cycling
                detected_at = (now - timedelta(days=days_ago)).replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )

                db.add(SpotOccupancyEvent(
                    spot_id=spot.id,
                    plate_text=plate_text,
                    detection_id=detection.id,
                    session_id=session_id,
                    detected_at=detected_at,
                    time_weight=_time_weight(detected_at),
                ))
                first_seen = detected_at if first_seen is None else min(first_seen, detected_at)
                last_seen = detected_at if last_seen is None else max(last_seen, detected_at)

            await db.flush()

            score, distinct_days = await _recompute_score(spot.id, plate_text, db)
            db.add(SpotOccupancyRecord(
                spot_id=spot.id,
                plate_text=plate_text,
                event_count=len(day_offsets),
                distinct_days=distinct_days,
                accumulated_score=score,
                first_seen_at=first_seen,
                last_seen_at=last_seen,
                is_flagged=score >= FLAG_THRESHOLD,
            ))
            await db.commit()

            print(
                f"  {spot_label} ← {plate_text:8s}  "
                f"score={score:5.2f}  events={len(day_offsets)}  "
                f"days={distinct_days}  flagged={score >= FLAG_THRESHOLD}"
            )


if __name__ == "__main__":
    asyncio.run(main())
