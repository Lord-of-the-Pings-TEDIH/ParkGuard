"""Session zone fallback — _resolve_zone_id picks the first ParkingZone when
no explicit zone_id is given and DEFAULT_ZONE_ID is unset."""

from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _zf_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def _zf_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _zf_bigint(type_, compiler, **kw):
    return "INTEGER"


from app.models.base import Base
from app.models.parking import ParkingZone
from app.routers.sessions import _resolve_zone_id

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


@pytest.mark.asyncio
async def test_explicit_zone_id_is_returned_unchanged(db: AsyncSession) -> None:
    assert await _resolve_zone_id(7, db) == 7


@pytest.mark.asyncio
async def test_fallback_to_first_zone_in_db(db: AsyncSession) -> None:
    zone = ParkingZone(name="Test Zone", code="ZF-01", grace_period_min=10, is_active=True)
    db.add(zone)
    await db.flush()

    with patch("app.routers.sessions.settings") as mock_settings:
        mock_settings.DEFAULT_ZONE_ID = None
        resolved = await _resolve_zone_id(None, db)

    assert resolved == zone.id


@pytest.mark.asyncio
async def test_returns_none_when_db_empty_and_no_default(db: AsyncSession) -> None:
    with patch("app.routers.sessions.settings") as mock_settings:
        mock_settings.DEFAULT_ZONE_ID = None
        resolved = await _resolve_zone_id(None, db)

    assert resolved is None


@pytest.mark.asyncio
async def test_default_zone_id_setting_takes_precedence(db: AsyncSession) -> None:
    with patch("app.routers.sessions.settings") as mock_settings:
        mock_settings.DEFAULT_ZONE_ID = 99
        resolved = await _resolve_zone_id(None, db)

    assert resolved == 99
