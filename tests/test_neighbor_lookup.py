"""X / X+1 neighbour enrichment — WRONG_PLATE SpotMatch carries owner_spot
when the interloper's plate owns a different spot in the same lot."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _nl_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def _nl_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _nl_bigint(type_, compiler, **kw):
    return "INTEGER"


from app.models.base import Base
from app.models.parking_lot import ParkingLot
from app.models.parking_spot import ParkingSpot
from app.services.parking_validation import validate_parking_at_location

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


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


@pytest_asyncio.fixture
async def two_adjacent_spots(db: AsyncSession):
    lot = ParkingLot(name="Neighbour Lot", total_spots=2)
    db.add(lot)
    await db.flush()

    spot1 = ParkingSpot(
        spot_label="A-01",
        parking_lot_id=lot.id,
        latitude=46.7700,
        longitude=23.5895,
        assigned_plate="B11AAA",
        spot_sequence=1,
    )
    # spot2 is ~5 m away — outside the default 40 m radius but same lot
    spot2 = ParkingSpot(
        spot_label="A-02",
        parking_lot_id=lot.id,
        latitude=46.7705,
        longitude=23.5895,
        assigned_plate="B22BBB",
        spot_sequence=2,
    )
    db.add(spot1)
    db.add(spot2)
    await db.flush()
    return spot1, spot2


@pytest.mark.asyncio
async def test_wrong_plate_has_owner_spot(db: AsyncSession, two_adjacent_spots) -> None:
    spot1, spot2 = two_adjacent_spots

    # B22BBB parked at spot1 (owns spot2) → WRONG_PLATE with owner_spot == spot2
    result = await validate_parking_at_location(
        target_lat=spot1.latitude,
        target_lon=spot1.longitude,
        plate_text="B22BBB",
        db=db,
    )

    assert result.status == "WRONG_PLATE"
    assert result.spot is not None and result.spot.id == spot1.id
    assert result.owner_spot is not None
    assert result.owner_spot.id == spot2.id


@pytest.mark.asyncio
async def test_match_has_no_owner_spot(db: AsyncSession, two_adjacent_spots) -> None:
    spot1, _ = two_adjacent_spots

    result = await validate_parking_at_location(
        target_lat=spot1.latitude,
        target_lon=spot1.longitude,
        plate_text="B11AAA",
        db=db,
    )

    assert result.status == "MATCH"
    assert result.owner_spot is None


@pytest.mark.asyncio
async def test_wrong_plate_no_owner_returns_none_owner_spot(
    db: AsyncSession, two_adjacent_spots
) -> None:
    spot1, _ = two_adjacent_spots

    # CJ05XYZ has no spot in this lot → owner_spot is None
    result = await validate_parking_at_location(
        target_lat=spot1.latitude,
        target_lon=spot1.longitude,
        plate_text="CJ05XYZ",
        db=db,
    )

    assert result.status == "WRONG_PLATE"
    assert result.owner_spot is None
