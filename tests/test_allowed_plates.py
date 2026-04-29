"""Co-use whitelist — a plate in allowed_plates yields MATCH, not WRONG_PLATE."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _ap_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def _ap_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _ap_bigint(type_, compiler, **kw):
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
async def spot_with_allowed(db: AsyncSession) -> ParkingSpot:
    lot = ParkingLot(name="Test Lot", total_spots=1)
    db.add(lot)
    await db.flush()
    spot = ParkingSpot(
        spot_label="A-01",
        parking_lot_id=lot.id,
        latitude=46.7700,
        longitude=23.5895,
        assigned_plate="B11AAA",
        allowed_plates=["B22BBB"],
    )
    db.add(spot)
    await db.flush()
    return spot


@pytest.mark.asyncio
async def test_assigned_plate_matches(db: AsyncSession, spot_with_allowed: ParkingSpot) -> None:
    result = await validate_parking_at_location(
        target_lat=spot_with_allowed.latitude,
        target_lon=spot_with_allowed.longitude,
        plate_text="B11AAA",
        db=db,
    )
    assert result.status == "MATCH"


@pytest.mark.asyncio
async def test_allowed_plate_matches(db: AsyncSession, spot_with_allowed: ParkingSpot) -> None:
    result = await validate_parking_at_location(
        target_lat=spot_with_allowed.latitude,
        target_lon=spot_with_allowed.longitude,
        plate_text="B22BBB",
        db=db,
    )
    assert result.status == "MATCH"


@pytest.mark.asyncio
async def test_unlisted_plate_is_wrong(db: AsyncSession, spot_with_allowed: ParkingSpot) -> None:
    result = await validate_parking_at_location(
        target_lat=spot_with_allowed.latitude,
        target_lon=spot_with_allowed.longitude,
        plate_text="CJ05XYZ",
        db=db,
    )
    assert result.status == "WRONG_PLATE"
