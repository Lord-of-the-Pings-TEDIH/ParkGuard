"""cleanup_session_occupancy correctly unflags a record when removing the
events from the deleted session drops the score below FLAG_THRESHOLD."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _cu_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def _cu_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _cu_bigint(type_, compiler, **kw):
    return "INTEGER"


from app.models.base import Base
from app.models.parking_lot import ParkingLot
from app.models.parking_spot import ParkingSpot
from app.models.occupancy import SpotOccupancyEvent, SpotOccupancyRecord
from app.services.occupancy import FLAG_THRESHOLD, cleanup_session_occupancy

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

UTC = timezone.utc


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.mark.asyncio
async def test_cleanup_unflags_record(db: AsyncSession) -> None:
    lot = ParkingLot(name="CU Lot", total_spots=1)
    db.add(lot)
    await db.flush()

    spot = ParkingSpot(
        spot_label="A-01",
        parking_lot_id=lot.id,
        latitude=46.77,
        longitude=23.59,
        assigned_plate="B11AAA",
    )
    db.add(spot)
    await db.flush()

    session_id = uuid.uuid4()
    plate = "CJ05XYZ"
    now = datetime.now(UTC)

    # Seed enough events to produce a score above FLAG_THRESHOLD.
    # Each event at 03:00 (night weight=1.5); decay≈1.0 for recent.
    # 10 events × 1.5 = 15 > FLAG_THRESHOLD (8.0)
    for i in range(10):
        db.add(SpotOccupancyEvent(
            spot_id=spot.id,
            plate_text=plate,
            detected_at=now.replace(hour=3) - timedelta(hours=i),
            time_weight=1.5,
            session_id=session_id,
        ))
    await db.flush()

    # Manually create a flagged record matching the seeded events.
    record = SpotOccupancyRecord(
        spot_id=spot.id,
        plate_text=plate,
        event_count=10,
        distinct_days=1,
        accumulated_score=FLAG_THRESHOLD + 5.0,
        first_seen_at=now,
        last_seen_at=now,
        is_flagged=True,
    )
    db.add(record)
    await db.flush()

    # Remove all events from the session → score drops to 0 → unflagged.
    await cleanup_session_occupancy(session_id, db)

    result = await db.execute(
        select(SpotOccupancyRecord)
        .where(SpotOccupancyRecord.spot_id == spot.id)
        .where(SpotOccupancyRecord.plate_text == plate)
    )
    updated = result.scalars().first()
    assert updated is None or updated.is_flagged is False


@pytest.mark.asyncio
async def test_cleanup_preserves_record_when_events_remain(db: AsyncSession) -> None:
    lot = ParkingLot(name="CU Lot 2", total_spots=1)
    db.add(lot)
    await db.flush()

    spot = ParkingSpot(
        spot_label="B-01",
        parking_lot_id=lot.id,
        latitude=46.78,
        longitude=23.59,
        assigned_plate="B11AAA",
    )
    db.add(spot)
    await db.flush()

    session_a = uuid.uuid4()
    session_b = uuid.uuid4()
    plate = "CJ07XYZ"
    now = datetime.now(UTC)

    # Session A: 3 events
    for i in range(3):
        db.add(SpotOccupancyEvent(
            spot_id=spot.id,
            plate_text=plate,
            detected_at=now - timedelta(hours=i),
            time_weight=1.0,
            session_id=session_a,
        ))
    # Session B: 10 events (enough to keep score above threshold after A removed)
    for i in range(10):
        db.add(SpotOccupancyEvent(
            spot_id=spot.id,
            plate_text=plate,
            detected_at=now - timedelta(hours=24 + i),
            time_weight=1.5,
            session_id=session_b,
        ))
    await db.flush()

    record = SpotOccupancyRecord(
        spot_id=spot.id,
        plate_text=plate,
        event_count=13,
        distinct_days=2,
        accumulated_score=FLAG_THRESHOLD + 5.0,
        first_seen_at=now - timedelta(days=1),
        last_seen_at=now,
        is_flagged=True,
    )
    db.add(record)
    await db.flush()

    await cleanup_session_occupancy(session_a, db)

    result = await db.execute(
        select(SpotOccupancyRecord)
        .where(SpotOccupancyRecord.spot_id == spot.id)
        .where(SpotOccupancyRecord.plate_text == plate)
    )
    updated = result.scalars().first()
    assert updated is not None
    assert updated.event_count == 10
