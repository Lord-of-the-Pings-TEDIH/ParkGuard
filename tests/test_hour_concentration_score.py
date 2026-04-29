"""Hour-concentration multiplier — _recompute_score applies a 1.3× multiplier
when >= 5 events cluster at the same hour (concentration >= 0.7)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _hc_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def _hc_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _hc_bigint(type_, compiler, **kw):
    return "INTEGER"


from app.models.base import Base
from app.models.parking_lot import ParkingLot
from app.models.parking_spot import ParkingSpot
from app.models.occupancy import SpotOccupancyEvent
from app.services.occupancy import _recompute_score

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


async def _make_spot(db: AsyncSession) -> ParkingSpot:
    lot = ParkingLot(name="HC Lot", total_spots=1)
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
    return spot


def _make_event(spot_id: int, hour: int, days_ago: int = 0) -> SpotOccupancyEvent:
    now = datetime.now(UTC)
    detected = now.replace(hour=hour, minute=0, second=0, microsecond=0) - timedelta(days=days_ago)
    return SpotOccupancyEvent(
        spot_id=spot_id,
        plate_text="CJ05XYZ",
        detected_at=detected,
        time_weight=1.5,  # night weight
    )


@pytest.mark.asyncio
async def test_concentrated_events_get_multiplier(db: AsyncSession) -> None:
    spot = await _make_spot(db)
    # 5 events all at 03:00 → high concentration → multiplier applies
    for i in range(5):
        db.add(_make_event(spot.id, hour=3, days_ago=i))
    await db.flush()

    score, _, conc, _ = await _recompute_score(spot.id, "CJ05XYZ", db)

    # Without multiplier: base ≈ 5 × 1.5 × ~1.0 ≈ 7.5
    # With 1.3× multiplier: base ≈ 9.75
    assert conc is not None and conc >= 0.95, f"expected high concentration, got {conc}"
    assert score > 9.0, f"expected score > 9.0 (multiplier applied), got {score}"


@pytest.mark.asyncio
async def test_spread_events_no_multiplier(db: AsyncSession) -> None:
    spot = await _make_spot(db)
    # 5 events at very different hours → low concentration → no multiplier
    for i, hour in enumerate([1, 7, 13, 19, 22]):
        db.add(_make_event(spot.id, hour=hour, days_ago=i))
    await db.flush()

    score, _, conc, _ = await _recompute_score(spot.id, "CJ05XYZ", db)

    # Concentration is below 0.7 threshold → no multiplier
    # Score is roughly Σ time_weight × decay ≈ ≤ 9.0
    assert score <= 9.0, f"expected no multiplier for spread events, got score={score}, conc={conc}"


@pytest.mark.asyncio
async def test_fewer_than_min_events_no_multiplier(db: AsyncSession) -> None:
    spot = await _make_spot(db)
    # Only 4 events at the same hour — below OCCUPANCY_HOUR_CONCENTRATION_MIN_EVENTS=5
    for i in range(4):
        db.add(_make_event(spot.id, hour=3, days_ago=i))
    await db.flush()

    score, _, conc, _ = await _recompute_score(spot.id, "CJ05XYZ", db)

    # Min events gate not cleared → multiplier not applied
    # base ≈ 4 × 1.5 = 6.0 without multiplier
    assert score < 8.5, f"expected no multiplier for <5 events, got score={score}"
