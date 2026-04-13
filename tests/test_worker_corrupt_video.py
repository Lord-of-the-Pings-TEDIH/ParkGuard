"""Worker error-handling: a corrupt/missing video must mark the session failed."""

from __future__ import annotations

import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import BigInteger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB, UUID

# SQLite shims for Postgres-only column types
@compiles(JSONB, "sqlite")
def _compile_jsonb(type_, compiler, **kw):
    return "TEXT"

@compiles(UUID, "sqlite")
def _compile_uuid(type_, compiler, **kw):
    return "CHAR(32)"

@compiles(BigInteger, "sqlite")
def _compile_bigint(type_, compiler, **kw):
    return "INTEGER"

from app.models.base import Base
from app.models.detection import Session as SessionRow
from app.models import parking  # noqa: F401  — register tables
from app.pipeline import worker

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


def _db_factory():
    return TestingSessionLocal()


@pytest.mark.asyncio
async def test_corrupt_video_marks_session_failed(db: AsyncSession, tmp_path):
    # Seed a session row
    session_id = uuid.uuid4()
    db.add(SessionRow(id=session_id, source_filename="bad.mp4", status="pending"))
    await db.commit()

    # Corrupt video file: not a real video container
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_bytes(b"this is not a video")

    # Reset semaphore in case prior tests left it locked
    worker.PROCESSING_SEMAPHORE = __import__("asyncio").Semaphore(1)

    await worker.process_video(
        session_id=session_id,
        video_path=str(corrupt),
        fps_target=5,
        db_factory=_db_factory,
    )

    # Re-fetch in a fresh session
    async with TestingSessionLocal() as fresh:
        row = await fresh.get(SessionRow, session_id)
        assert row is not None
        assert row.status == "failed"
        assert row.error_message
        assert len(row.error_message) <= 500
