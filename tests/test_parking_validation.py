"""Tests for app.services.parking_validation.validate_parking_at_location."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb(type_, compiler, **kw):  # noqa: ARG001
    return "TEXT"


@compiles(UUID, "sqlite")
def _compile_uuid(type_, compiler, **kw):  # noqa: ARG001
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _compile_bigint(type_, compiler, **kw):  # noqa: ARG001
    return "INTEGER"


from app.models.base import Base
from app.models.parking_lot import ParkingLot
from app.models.parking_spot import ParkingSpot
from app.services.parking_validation import (
    DEFAULT_SEARCH_RADIUS_M,
    validate_parking_at_location,
)


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


async def _seed_lot_with_spot(
    db: AsyncSession,
    *,
    lat: float,
    lon: float,
    plate: str | None,
    label: str = "A-01",
) -> ParkingSpot:
    lot = ParkingLot(name="Test Lot", total_spots=1)
    db.add(lot)
    await db.flush()
    spot = ParkingSpot(
        spot_label=label,
        parking_lot_id=lot.id,
        latitude=lat,
        longitude=lon,
        assigned_plate=plate,
    )
    db.add(spot)
    await db.flush()
    return spot


@pytest.mark.asyncio
async def test_match_when_plate_equals_assigned(db: AsyncSession) -> None:
    spot = await _seed_lot_with_spot(db, lat=46.7700, lon=23.5895, plate="B11AAA")

    result = await validate_parking_at_location(
        target_lat=spot.latitude,
        target_lon=spot.longitude,
        plate_text="B 11 AAA",
        db=db,
    )

    assert result.status == "MATCH"
    assert result.spot is not None
    assert result.spot.id == spot.id
    assert result.distance_m is not None and result.distance_m < 1.0


@pytest.mark.asyncio
async def test_wrong_plate_when_someone_else_parked_in_spot(db: AsyncSession) -> None:
    spot = await _seed_lot_with_spot(db, lat=46.7700, lon=23.5895, plate="B11AAA")

    result = await validate_parking_at_location(
        target_lat=spot.latitude,
        target_lon=spot.longitude,
        plate_text="CJ99XYZ",
        db=db,
    )

    assert result.status == "WRONG_PLATE"
    assert result.spot is not None and result.spot.id == spot.id


@pytest.mark.asyncio
async def test_unreserved_spot_excluded_from_match(db: AsyncSession) -> None:
    """An unreserved (assigned_plate=NULL) spot must not produce a verdict.

    The validator filters out unreserved spots in the SQL prefilter so they
    cannot trigger WRONG_PLATE.  Real-world reasoning: a spot with no owner
    has no one to wrong, and feeding such "verdicts" to the suspicion-scoring
    pipeline would generate noise on free public bays.
    """
    spot = await _seed_lot_with_spot(db, lat=46.7700, lon=23.5895, plate=None)

    result = await validate_parking_at_location(
        target_lat=spot.latitude,
        target_lon=spot.longitude,
        plate_text="B11AAA",
        db=db,
    )

    assert result.status == "NO_SPOT_FOUND"
    assert result.spot is None


@pytest.mark.asyncio
async def test_no_spot_found_outside_radius(db: AsyncSession) -> None:
    await _seed_lot_with_spot(db, lat=46.7700, lon=23.5895, plate="B11AAA")

    # Roughly 50 m east — well outside the 3 m default radius.
    result = await validate_parking_at_location(
        target_lat=46.7700,
        target_lon=23.5901,
        plate_text="B11AAA",
        db=db,
    )

    assert result.status == "NO_SPOT_FOUND"
    assert result.spot is None
    assert result.distance_m is None


@pytest.mark.asyncio
async def test_nearest_of_two_spots_is_chosen(db: AsyncSession) -> None:
    near = await _seed_lot_with_spot(
        db, lat=46.770000, lon=23.589500, plate="CJ05XYZ", label="A-01"
    )
    # Second spot ~6 m north of the first; both are inside the 7 m radius
    # we use here, but `near` should win because it's closer.
    far_lat = near.latitude + (6.0 / 111_320.0)
    far = ParkingSpot(
        spot_label="A-02",
        parking_lot_id=near.parking_lot_id,
        latitude=far_lat,
        longitude=near.longitude,
        assigned_plate="B22BBB",
    )
    db.add(far)
    await db.flush()

    result = await validate_parking_at_location(
        target_lat=near.latitude,
        target_lon=near.longitude,
        plate_text="CJ05XYZ",
        db=db,
        radius_m=7.0,
    )

    assert result.status == "MATCH"
    assert result.spot is not None and result.spot.id == near.id


@pytest.mark.asyncio
async def test_canonicalization_strips_spaces_and_dashes(db: AsyncSession) -> None:
    spot = await _seed_lot_with_spot(db, lat=46.7700, lon=23.5895, plate="b-11 aaa")

    result = await validate_parking_at_location(
        target_lat=spot.latitude,
        target_lon=spot.longitude,
        plate_text="B11AAA",
        db=db,
    )

    assert result.status == "MATCH"


@pytest.mark.asyncio
async def test_radius_must_be_positive(db: AsyncSession) -> None:
    with pytest.raises(ValueError):
        await validate_parking_at_location(
            target_lat=0.0,
            target_lon=0.0,
            plate_text="X",
            db=db,
            radius_m=0.0,
        )


def test_default_radius_matches_smartphone_gps_uncertainty() -> None:
    # 40 m covers the realistic worst case of consumer GPS in urban canyons
    # (5–15 m open-sky, 20–40 m dense block).  Tightening this back to a few
    # metres breaks every spot-match in real footage; loosening past 50 m
    # starts producing cross-street false matches in dense neighbourhoods.
    assert DEFAULT_SEARCH_RADIUS_M == 40.0
